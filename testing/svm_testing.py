import os
import joblib

# Resolve project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define model directory
MODEL_DIR = os.path.join(BASE_DIR, "models", "svm_baseline")

# Load the TF-IDF vectoriser and trained SVM model
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
svm_model = joblib.load(os.path.join(MODEL_DIR, "linear_svm.joblib"))

while True:
    review = input("\nEnter a product review (or type 'exit' to quit): ").strip()

    if review.lower() == "exit":
        break

    # Transform the input text and run prediction
    X = vectorizer.transform([review])
    pred_label = svm_model.predict(X)[0]

    print("Predicted sentiment:", pred_label)