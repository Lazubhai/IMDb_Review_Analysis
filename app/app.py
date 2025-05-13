from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

feedback_file = 'data/feedback_data.csv'

@app.route('/')
def home():
    return render_template('index.html')  # Make sure you have this template (index.html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data['review']
        review_vector = vectorizer.transform([review])
        sentiment = model.predict(review_vector)
        sentiment_label = 'positive' if sentiment == 1 else 'negative'
        return jsonify({'sentiment': sentiment_label})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        review = data['review']
        sentiment = data['sentiment']
        feedback_data = {'review': [review], 'sentiment': [sentiment]}
        feedback_df = pd.DataFrame(feedback_data)

        if os.path.exists(feedback_file):
            feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_file, index=False)

        return jsonify({'message': 'Feedback saved successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        if not os.path.exists(feedback_file):
            return jsonify({'error': 'No feedback data found to retrain the model.'})

        feedback_df = pd.read_csv(feedback_file)

        if len(feedback_df) == 0:
            return jsonify({'error': 'No feedback data to train on.'})

        X_feedback = feedback_df['review']
        y_feedback = feedback_df['sentiment']
        X_feedback_vec = vectorizer.transform(X_feedback)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_feedback_vec, y_feedback)

        joblib.dump(model, 'model/sentiment_model.pkl')
        joblib.dump(vectorizer, 'model/vectorizer.pkl')

        os.remove(feedback_file)

        return jsonify({'message': 'Model retrained successfully with new feedback data!'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
