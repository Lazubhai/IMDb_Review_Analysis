# src/predict_sentiment.py

import joblib  # for loading the model and vectorizer

# Function to predict sentiment of a new review
def predict_sentiment(review):
    # Load the saved model and vectorizer
    model = joblib.load('models/sentiment_model.pkl')  # Adjust path if necessary
    vectorizer = joblib.load('models/vectorizer.pkl')  # Adjust path if necessary

    # Transform the review using the vectorizer
    review_vector = vectorizer.transform([review])

    # Predict the sentiment (0 = Negative, 1 = Positive)
    sentiment = model.predict(review_vector)

    # Return sentiment as 'positive' or 'negative'
    return 'positive' if sentiment == 1 else 'negative'

# Test with a sample review
if __name__ == "__main__":
    new_review = "This movie was fantastic! I loved it."
    print(f"Sentiment: {predict_sentiment(new_review)}")
