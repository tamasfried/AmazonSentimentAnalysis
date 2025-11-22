# Amazon Sentiment Analysis

This repository contains my MSc project for multi-class sentiment analysis of Amazon product reviews. The goal is to compare classical machine learning, deep learning and transformer-based approaches to classify reviews as **negative**, **neutral**, or **positive**.

## Project Overview

This project uses the **Amazon Fine Food Reviews** dataset, sourced from Kaggle:

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

The raw dataset contains over 500,000 Amazon product reviews.  
For practical experimentation, I created a **balanced subset of 120,000 reviews** (40,000 per class) which is cleaned and ready for modelling.

## Current Progress

- Loaded and explored the full dataset  
- Converted 1–5 star ratings into 3 sentiment classes  
- Created an even 40k-per-class balanced dataset  
- Cleaned text (lowercasing, removing newlines and whitespace)  
- Saved the processed dataset as `balanced_reviews.csv`  
- Documented all steps in `explore_data.ipynb`  
- Set up the repository structure and environment for modelling

## Next Steps

- Build a TF-IDF + LinearSVC baseline classifier  
- Develop deep-learning models (LSTM, CNN‑LSTM)  
- Fine‑tune transformer models (BERT, DistilBERT)  
- Evaluate all models using accuracy, precision, recall and F1‑score  
- Create a simple Python artefact for running predictions on new text  
- Prepare results and discussion for the dissertation

## Repository Structure

```
.
├── data/
│   ├── balanced_reviews.csv      # Cleaned & balanced dataset (small enough for GitHub)
│   └── FineFoodReviews.csv       # Raw dataset (ignored via .gitignore)
│
├── notebooks/
│   └── explore_data.ipynb        # Data exploration, cleaning, balancing
│
├── src/                          # Training scripts (to be added)
│
├── .gitignore
└── README.md
```

## Notes

- The raw dataset is **not** tracked in Git due to GitHub’s 100MB limit.
- The project supports both experimentation and the artefact required for the MSc dissertation.