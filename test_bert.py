import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from debug_module import mock_backend
import os

# Setup
MODEL_ID = "google-bert/bert-base-multilingual-cased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def build_inputs(tokenizer, device):
    lines = [
        "Paris is the [MASK] of France.",
        "París es la [MASK] de Francia.",
    ]
    return tokenizer(lines, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)

def run_bert_test():
    print("\n=== Loading BERT Model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).eval().to(device)
    
    batch = build_inputs(tokenizer, device)
    
    print("\n=== Compiling with Mock Backend ===")
    # We expect this to succeed if BERT uses only supported ops/dtypes (mostly float32/int64)
    # Note: BERT might use LayerNorm or Gelu which we haven't banned yet.
    compiled_model = torch.compile(model, backend=mock_backend)
    
    print("\n=== Running Inference ===")
    with torch.no_grad():
        try:
            output = compiled_model(**batch)
            print("Inference Successful!")
            print("Logits shape:", output.logits.shape)
        except Exception as e:
            print("Inference Failed!")
            print(e)
            raise e

    # Check for artifacts
    artifact_dir = "debug_artifacts"
    if os.path.exists(artifact_dir) and os.listdir(artifact_dir):
        print(f"\n[PASS] Artifacts generated in {artifact_dir}:")
        for f in os.listdir(artifact_dir):
            print(f" - {f}")
    else:
        print("\n[FAIL] No artifacts found!")

if __name__ == "__main__":
    run_bert_test()
