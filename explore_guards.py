import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch._dynamo

# Setup
MODEL_ID = "google-bert/bert-base-multilingual-cased"
device = torch.device("cpu") # Keep it simple for exploration

def build_inputs(tokenizer):
    lines = ["Paris is the [MASK] of France."]
    return tokenizer(lines, return_tensors="pt", padding=True, truncation=True, max_length=32)

def explore():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).eval()
    inputs = build_inputs(tokenizer)
    
    print("Running torch._dynamo.explain...")
    # explain returns an ExplainOutput object
    explanation = torch._dynamo.explain(model, **inputs)
    
    print("\n=== Explanation Object Attributes ===")
    print(dir(explanation))
    return


if __name__ == "__main__":
    explore()
