# Amazon-Product-Reviews-Sentiment-Analysis
This Python script analyzes Amazon product reviews using natural language processing (NLP) techniques to perform sentiment analysis and recommend similar products based on cosine similarity.

## Requirements
- Python 3.x
- Libraries: numpy, pandas, spacy, scikit-learn
## Installation
1. Clone the repository or download the script.
2. Install the required libraries:

```bash
  pip install numpy pandas spacy scikit-learn
```
3. Download the English language model for spaCy:
``` bash
   python -m spacy download en_core_web_md
  ```
4. Place the amazon_product_reviews.csv file in the same directory as the script.

## Usage
1. Run the script 'sentiment_analysis.py'.

```bash
python sentiment_analysis.py
```
2. The script performs the following steps:

- Loads the Amazon product reviews dataset from amazon_product_reviews.csv.
- Performs simple exploratory data analysis (EDA) to check the dataset's structure.
- Cleans the data by removing rows with missing values in the 'reviews.text' column.
- Preprocesses the text data by lemmatizing, removing stopwords, and punctuation.
- Computes vector representations of the preprocessed text using spaCy's word embeddings.
- Defines a function sentiment_analysis to perform sentiment analysis and recommend similar products based on input reviews.
- Tests the sentiment_analysis function using sample reviews.
3. Modify the `testing_reviews` list to test the sentiment analysis function with your own reviews and adjust the number of recommendations (num_recommendations) as needed.
