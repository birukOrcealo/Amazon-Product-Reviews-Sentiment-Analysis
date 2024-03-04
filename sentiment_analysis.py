import numpy as np
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import spacy 
df=pd.read_csv('amazon_product_reviews.csv',low_memory=False)
df.head()
#simple EDA
print(f'columns: {df.columns}')
print(df.shape())# number of rows and columns 
print(df['reviews.rating'])

#Data Cleaning 
print(f'number of null values in the reviews.text column :{df['reviews.text'].isnull().sum()}') #check for missing data in 'reviews.txt' column
df.dropna(subset=['reviews.text'],inplace=True)# remove the null values from  reviews.txt column 
print(df['reviews.text'].isnull().sum())# re check for missing data in 'reviews.txt' column
df.columns

# Load the English language model
nlp=spacy.load('en_core_web_md')
# Function to preprocess text
def preprocess(text):
    # Tokenize text with NLTK
    cm=nlp(text) 
    # Lemmatize and filter out stopwords and punctuation
    return ''.join([token.lemma_ # Lemmatize
                    for token in cm # Tokenize
                    if not token.is_stop  # Filter stopwords
                    and not token.is_punct])  # Filter punctuation
df['preprocessed_review']=df['reviews.text'].apply(preprocess) #  Apply the function to all review.text rows and creat 'preprocessed_review' column 
print(f'check if preprocessed_review column is created successfully :{df.columns}')

# Function to get vector of a text using nlp
def get_vector(text):
    # Initialize the nlp object 
    doc=nlp(text)
    # Return the vector of the document
    return doc.vector
#  Apply the function to all review.text rows and creat 'preprocessed_review' column 
df['vector_col']= df['preprocessed_review'].apply(get_vector)
print(f'check if vector_col is created successfully : {df['vector_col']}') 



def sentiment_analysis(input_review, n_recommendations):
    # Get vector from input review
    input_vector = get_vector(preprocess(input_review))
    input_vector = input_vector.reshape(1,-1)  # Reshape input_vector to be a row vector

    # Convert the DataFrame column to a numpy array
    vector_col_array = np.array(df['vector_col'].to_list())

    # Compute cosine similarity between input_vector and vector_col_array
    similarity_score = cosine_similarity(input_vector, vector_col_array)

    # Get top N similarity
    similarity_index = similarity_score[0].argsort()[-n_recommendations-1:-1][::-1]

    # Get rating based on the indices
    recommended_ratings = df['reviews.rating'].iloc[similarity_index].tolist()

    return recommended_ratings

# Testing function
testing_reviews = [
    "This product exceeded my expectations. It's amazing!",
    "I'm extremely satisfied with my purchase. The quality is top-notch.",
    "I wish I had bought this product sooner. It's a game-changer!",
    "Not impressed with the product. It didn't meet my expectations.",
    "This is the worst product I've ever purchased. It's a complete waste of money."
]
num_recommendations = 1

for review in testing_reviews:
    ratings = sentiment_analysis(review, num_recommendations)
    print(ratings)
