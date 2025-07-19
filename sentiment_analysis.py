# Capstone Project: Sentiment Analysis on Amazon Product Reviews

# Import the relevant python libraries/packages to begin the program
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import re

# Load spaCy model with TextBlob intergration
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')
# Load dataset
data = pd.read_csv(r"C:\Users\jeffi\Downloads\amazon_product_reviews.csv")

# Function to preprocess text data
def preprocess_text(text):
    # Handle missing values by imputing with an empty string
    if pd.isnull(text):
        return ''
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation using regular expressions
    text = re.sub(r'[^\w\s]', '', text)
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Remove stopwords and non-alphabetic characters, and lemmatize remaining characters
    processed_text = [char.lemma_ for char in doc if not char.is_stop and char.is_alpha]
    
    # Join processed characters into a string
    return ' '.join(processed_text)

# Preprocess text data
data['processed_text'] = data['reviews.text'].apply(preprocess_text)

# To remove all missing values from the 'reviews.text' column, use the dropna() function from pandas
clean_data = data.dropna(subset=['reviews.text'])

# Function for sentiment analysis
def predict_sentiment(review):
    # Process the review text with spaCy
    doc = nlp(review)
    
    # Access TextBlob sentiment analysis results
    sentiment_score = doc._.sentiment.polarity
    
    # Classify sentiment based on sentiment score
    if sentiment_score > 0:
        return 'Positive'
    elif sentiment_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Preprocess text data
data['processed_text'] = data['reviews.text'].apply(preprocess_text)

# Perform sentiment analysis
data['sentiment'] = data['processed_text'].apply(predict_sentiment)

# Drop rows with missing values in the 'processed_text' column (if needed)
data.dropna(subset=['processed_text'], inplace=True)

# Optional: Drop original text column if not needed
data.dropna(columns=['reviews.text'], inplace=True)

# Display the preprocessed data with sentiment analysis results
print(data.head())
