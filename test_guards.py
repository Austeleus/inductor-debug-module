import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from debug_module import GuardInspector

# Setup
MODEL_ID = "google-bert/bert-base-multilingual-cased"

def test_guard_inspector():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).eval()
    
    lines = ["Paris is the [MASK] of France."]
    inputs = tokenizer(lines, return_tensors="pt", padding=True, truncation=True, max_length=32)
    
    print("Running Guard Inspector...")
    inspector = GuardInspector(model)
    report = inspector.inspect(inputs)
    
    inspector.print_report(report)

if __name__ == "__main__":
    test_guard_inspector()
