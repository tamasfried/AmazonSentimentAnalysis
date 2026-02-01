# Prompt user to enter a review to manually test the fine-tuned DistilBERT model

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "../models/distilbert_sentiment"

# Load tokenizer and model from local folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load label mapping
with open(f"{MODEL_DIR}/label_map.json", "r") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

while True:
    # Prompt for user input
    review = input("\nEnter a product review (or type 'exit' to quit): ").strip()

    if review.lower() == "exit":
        break

    # Tokenise the input text
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Run inference
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # Extract prediction
    pred_id = int(torch.argmax(probs).item())

    print("Predicted sentiment:", label_map[pred_id])
    print("Confidence:", round(float(probs[pred_id].item()), 4))