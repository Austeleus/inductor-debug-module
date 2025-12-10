# mbert_mock_benchmark.py

import os, time, torch, subprocess
from transformers import AutoTokenizer, AutoModelForMaskedLM
from debug_module import mock_backend

WARMUP, ITERS, PRINT_EVERY = 3, 10, 2
MODEL_ID = "google-bert/bert-base-multilingual-cased"
VERBOSE_DYNAMO = False

os.environ["TORCH_LOGS"] = "+dynamo,graph_breaks" if VERBOSE_DYNAMO else "graph_breaks"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

def log(msg): print(msg, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"[info] device: {device}")
if device.type == "cuda":
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ).decode().strip()
        log("[gpu]\n" + out)
    except Exception:
        pass

def build_inputs(tokenizer, device):
    lines = [
        "Paris is the [MASK] of France.",
        "París es la [MASK] de Francia.",
        "Paris ist die [MASK] von Frankreich.",
        "पेरिस [MASK] का राजधानी है.",
    ]

    return tokenizer(
        lines,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=32,
    ).to(device)

@torch.no_grad()
def run_with_progress(model, batch, warmup, iters, label):
    if warmup > 0:
        log(f"[{label}] warmup: {warmup} iters")
    for i in range(warmup):
        _ = model(**batch)
        if (i + 1) % PRINT_EVERY == 0 or (i + 1) == warmup:
            log(f"[{label}] warmup {i+1}/{warmup}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    log(f"[{label}] timed: {iters} iters")
    t0 = time.perf_counter()
    last = None
    for i in range(iters):
        last = model(**batch)
        if ((i + 1) % PRINT_EVERY == 0) or ((i + 1) == iters):
            log(f"[{label}] progress {i+1}/{iters} | elapsed {time.perf_counter()-t0:.2f}s")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return avg_ms, last

def top_pred_tokens(logits, tokenizer, batch):
    mask_id = tokenizer.mask_token_id
    probs = torch.softmax(logits, dim=-1)
    preds = []
    for i in range(probs.shape[0]):
        idxs = (batch["input_ids"][i] == mask_id).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            preds.append("<no-mask>")
            continue
        pos = int(idxs[0])
        top_id = int(torch.argmax(probs[i, pos]))
        preds.append(tokenizer.decode([top_id], skip_special_tokens=True))
    return preds

def main():
    log(f"[info] loading {MODEL_ID}…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).eval().to(device)

    batch = build_inputs(tok, device)

    eager_ms, eager_out = run_with_progress(model, batch, WARMUP, ITERS, "eager")
    log(f"[eager] avg latency: {eager_ms:.2f} ms")

    log("[mock] compiling with mock_backend (TorchInductor under the hood)…")
    t0c = time.perf_counter()
    compiled = torch.compile(
        model,
        backend=mock_backend,
        fullgraph=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0c
    log(f"[mock] compile (front-end + backend) took: {compile_time:.2f}s")

    mock_ms, mock_out = run_with_progress(compiled, batch, WARMUP, ITERS, "mock")
    log(f"[mock] avg latency: {mock_ms:.2f} ms")

    ep = top_pred_tokens(eager_out.logits, tok, batch)
    mp = top_pred_tokens(mock_out.logits, tok, batch)
    log("\nTop-1 predictions (eager | mock):")
    for i, (a, b) in enumerate(zip(ep, mp)):
        log(f"  {i}: {a} | {b} {'✓' if a==b else '≠'}")

    log("\nSummary:")
    log(f"  Eager avg latency : {eager_ms:.2f} ms")
    log(f"  Mock avg latency  : {mock_ms:.2f} ms")
    log(f"  Mock compile time : {compile_time:.2f} s")

if __name__ == "__main__":
    main()
