from flask import Flask, render_template, request, send_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import uuid
import pandas as pd

# Create Flask application
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Define local model path
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "distilbert_sentiment")

# Define upload/output folders
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Define label mapping
label_map = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text: str):
    # Tokenise input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()

    # Get prediction and confidence
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())

    return label_map[pred_id], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    single_result = None
    single_confidence = None
    review_text = ""

    csv_message = None
    download_url = None

    if request.method == "POST":
        action = request.form.get("action")

        # Single review prediction
        if action == "single":
            review_text = request.form.get("review_text", "").strip()

            if review_text:
                label, confidence = predict_sentiment(review_text)
                single_result = label
                single_confidence = confidence

        # CSV batch prediction
        if action == "csv":
            file = request.files.get("csv_file")

            if file is None or file.filename.strip() == "":
                csv_message = "No CSV file selected."
            elif not file.filename.lower().endswith(".csv"):
                csv_message = "Only CSV files are supported."
            else:
                unique_id = uuid.uuid4().hex
                upload_path = os.path.join(UPLOAD_DIR, f"{unique_id}.csv")
                output_path = os.path.join(OUTPUT_DIR, f"{unique_id}_predictions.csv")

                file.save(upload_path)

                # Load CSV
                df = pd.read_csv(upload_path)

                # Validate required column
                if "Review" not in df.columns:
                    csv_message = "CSV must contain a column named 'Review'."
                else:
                    sentiments = []
                    confidences = []

                    # Predict row-by-row
                    for text in df["Review"].fillna("").astype(str).tolist():
                        label, confidence = predict_sentiment(text)
                        sentiments.append(label)
                        confidences.append(confidence)

                    # Add prediction columns
                    df["predicted_sentiment"] = sentiments
                    df["confidence"] = confidences

                    # Save output CSV
                    df.to_csv(output_path, index=False)

                    csv_message = "CSV analysed successfully. Download is ready."
                    download_url = f"/download/{unique_id}"

    return render_template(
        "index.html",
        single_result=single_result,
        single_confidence=single_confidence,
        review_text=review_text,
        csv_message=csv_message,
        download_url=download_url
    )

@app.route("/download/<file_id>", methods=["GET"])
def download(file_id):
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}_predictions.csv")
    return send_file(output_path, as_attachment=True, download_name="predictions.csv")

if __name__ == "__main__":
    app.run(debug=True)