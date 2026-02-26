# Detecting Sarcasm in News Headlines using NLP

## Overview
Sarcasm detection is a notoriously difficult Natural Language Processing (NLP) task because it relies heavily on nuance, context, and phrasing rather than purely negative or positive words. This project aims to classify roughly 28,000 news headlines as either "Sarcastic" (*The Onion*) or "Serious" (*HuffPost*) to demonstrate how specific text preprocessing strategies can capture comedic intent.

## Methods
I used a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer paired with a Logistic Regression classifier. 
* **Sarcasm-Specific Cleaning:** Unlike standard sentiment analysis, I intentionally kept stop words and punctuation (like "!" and "?") because common words and structures are often essential to the structure of sarcastic headlines.
* **N-Grams:** I utilized an `ngram_range=(1,2)` to capture bigrams (like "Area Man" or "Yeah right"), which exploratory data analysis revealed to be strong contextual predictors of sarcasm. 

## Key Results
* The Logistic Regression model successfully differentiated between serious reporting and satirical headlines with high accuracy.
* Feature importance and frequency analysis revealed that specific bigrams were heavily weighted indicators of sarcasm. 
* Error analysis showed that the model struggles primarily with "World Knowledge"—it misclassifies satirical headlines that mimic serious journalistic tones because it lacks the real-world context to understand the absurdity of the event.

## How to Run
1. Clone this repository to your local machine.
2. Download the "News Headlines Dataset for Sarcasm Detection" (v2) from Kaggle.
3. Place the downloaded `Sarcasm_Headlines_Dataset_v2.json` file into a `data/` folder in the root of this repository.
4. Open `notebooks/main_analysis.ipynb` in Jupyter Notebook or JupyterLab.
5. Run the cells sequentially to reproduce the exploratory data analysis, text cleaning, model training, and visualizations.

## Requirements
The following packages are required to run this notebook:
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

To install the required packages, run:
`pip install -r requirements.txt`