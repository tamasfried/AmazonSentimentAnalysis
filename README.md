# Amazon Sentiment Analysis

This repository contains my MSc project for multi-class sentiment analysis of Amazon product reviews. The goal is to compare classical machine learning, deep learning and transformer-based approaches to classify reviews as **negative**, **neutral**, or **positive**.

## Project Overview

This project uses the **Amazon Fine Food Reviews** dataset, sourced from Kaggle:

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

The raw dataset contains over 500,000 Amazon product reviews.  
For practical experimentation, I created a **balanced subset of 120,000 reviews** (40,000 per class) which is cleaned and ready for modelling.

## Current Progress

- Loaded and explored the full Amazon reviews dataset  
- Converted 1–5 star ratings into 3 sentiment classes (negative, neutral, positive)  
- Created an even 40k-per-class balanced dataset  
- Cleaned text (lowercasing, whitespace and newline handling)  
- Saved the processed dataset as `balanced_reviews.csv`  
- Documented preprocessing and exploration in `explore_data.ipynb`  

- Built a TF-IDF + Linear SVM baseline classifier  
- Evaluated the SVM model using accuracy and macro F1-score  
- Saved the trained SVM model and TF-IDF vectoriser as reusable artefacts  

- Fine-tuned a DistilBERT transformer model for sentiment classification  
- Evaluated DistilBERT on the same held-out test set  
- Saved the fine-tuned DistilBERT model for reuse  

- Created standalone Python scripts to run predictions using:
  - the Linear SVM model
  - the DistilBERT model 

## Next Steps

- Create a dedicated analysis notebook to compare SVM and DistilBERT performance  
- Add confusion matrices and per-class performance analysis  
- Decide whether to include an LSTM-based deep learning model for comparison  
- Build a simple Flask web application using one trained model as the deployment artefact  
- Evaluate and justify model selection for deployment  
- Prepare results, analysis, and discussion sections for the dissertation

## Repository Structure

```
AmazonSentimentAnalysis/
├── .gitignore
├── data/
│   ├── .DS_Store
│   └── balanced_reviews.csv
├── notebooks/
│   ├── BERT_baseline.ipynb
│   ├── explore_data.ipynb
│   └── svm_baseline.ipynb
├── README.md
└── testing/
    ├── distilBERT_testing.py
    └── svm_testing.py
```

## Notes

- The raw dataset is **not** tracked in Git due to GitHub’s 100MB limit.
- The project supports both experimentation and the artefact required for the MSc dissertation.