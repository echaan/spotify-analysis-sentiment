---

# Spotify App Review Sentiment Analysis
---

## Project Overview

This project performs sentiment analysis on user reviews of the **Spotify: Music and Podcasts** application from the Google Play Store. The workflow includes **data scraping**, **text preprocessing**, **sentiment labeling** using a lexicon-based approach, **data visualization**, and **deep learning model training** (LSTM, CNN, and GRU).

The goal is to classify user reviews into **positive**, **negative**, and **neutral** sentiments and compare the performance of different neural network architectures.

---
## Table of Content
* [Project Overview](#project-overview)
* [Tech Stack](#tech-stack)
* [Repository Structure](#repository-structure)
* [Project Workflow](#project-workflow)
* [How to Run](#how-to-run)
---

## Tech Stack

**Languages & Frameworks**

* Python 3.x
* TensorFlow / Keras
* Scikit-learn

**Libraries & Tools**

* `google-play-scraper` for review scraping
* `pandas`, `numpy` for data processing
* `matplotlib`, `seaborn`, `wordcloud` for visualization
* `nltk`, `Sastrawi` for NLP preprocessing
* `sklearn.feature_extraction.text` (TF-IDF) for feature extraction

---

## Repository Structure

```
spotify-analysis-sentiment/
│
├── scrapping.ipynb              # Script for scraping Spotify app reviews from Google Play Store
├── model_train_notebook.ipynb   # Notebook for data preprocessing, sentiment labeling, model training, and evaluation
├── review_spotify.csv           # Dataset containing scraped Spotify user reviews
└── requirements.txt             # List of required Python dependencies
```

---

## Project Workflow

### 1. Data Scraping

* Reviews are scraped using the `google-play-scraper` library.
* The dataset consists of **10,000** most relevant reviews from the Indonesian version of the Spotify app (`com.spotify.music`).
* Output file: `review_spotify.csv`.

### 2. Data Cleaning & Preprocessing

* Duplicate removal, text cleaning, case folding, slang normalization, tokenization, stopword removal, and stemming using **Sastrawi**.
* The preprocessing pipeline standardizes text for Indonesian and English words.

### 3. Sentiment Labeling

* Lexicon-based sentiment scoring using **Indonesian positive and negative word dictionaries**.
* Reviews are labeled as **positive**, **negative**, or **neutral** based on cumulative word scores.

### 4. Exploratory Data Analysis

* Visualization of sentiment distribution (pie chart, histogram).
* Word cloud generation and top frequent word analysis using TF-IDF.

### 5. Feature Extraction

* Text converted into padded sequences using **Keras Tokenizer**.
* Sentiment labels are encoded and one-hot encoded for deep learning models.

### 6. Model Training

Three deep learning architectures were implemented:

| Model | Training Split | Test Split | Test Accuracy | Loss |
| ----- | -------------- | ---------- | ------------- | ---- |
| LSTM  | 80%            | 20%        | 0.93          | 0.26 |
| CNN   | 70%            | 30%        | 0.90          | 0.31 |
| GRU   | 90%            | 10%        | 0.92          | 0.28 |

Each model was trained using categorical cross-entropy loss and the Adam optimizer.

### 7. Model Evaluation

* Performance is evaluated based on **accuracy** and **loss**.
* Results indicate that the **LSTM model** achieved the best overall performance.

### 8. Prediction

The trained models were tested on new sample reviews.
Example:

| Review                                                      | LSTM Prediction | CNN Prediction | GRU Prediction |
| ----------------------------------------------------------- | --------------- | -------------- | -------------- |
| “Spotify luar biasa! Koleksi lagu lengkap, audio jernih...” | positive        | positive       | positive       |
| “Spotify jelek! Iklan banyak, sering crash!”                | negative        | negative       | negative       |

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/echaan/spotify-analysis-sentiment.git
cd spotify-analysis-sentiment
```

### 2. Set Up Environment

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks

Execute the Jupyter Notebooks in sequence:

1. `scraping/spotify_review_scraper.ipynb` — to collect the dataset
2. `model/sentiment_analysis_spotify.ipynb` — to preprocess, analyze, and train models

### 5. Predict New Reviews

You can modify the `new_texts` list in the model notebook to test new Spotify reviews.

---
