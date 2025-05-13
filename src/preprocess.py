import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)           # remove HTML tags
    text = re.sub(r'http\S+', '', text)         # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)        # keep only letters and spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_reviews(input_file, output_folder):
    df = pd.read_csv(input_file)

    if 'text' not in df.columns or 'sentiment' not in df.columns:
        print("❌ Dataset must have 'text' and 'sentiment' columns.")
        return

    # Apply cleaning on 'text' column
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['cleaned_text'])

    # Save processed file
    output_file = os.path.join(output_folder, 'processed_test_reviews.csv')
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to {output_file}")

# Run the cleaner
preprocess_reviews(
    'A:\\IMDb_Review_Analysis\\data\\test.csv',
    'A:\\IMDb_Review_Analysis\\data'
)
