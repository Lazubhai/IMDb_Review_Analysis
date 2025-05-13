# random_forest_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset (Assuming 'data/train.csv')
data = pd.read_csv('data/test.csv')

# Assuming your dataset has 'text' (reviews) and 'label' (sentiment) columns
X = data['text']
y = data['sentiment']

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=2)

# Transform the text data to numerical features using TF-IDF
X_tfidf = vectorizer.fit_transform(X)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier with 200 trees
model_rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the Random Forest model
model_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_rf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Save the trained model and the TF-IDF vectorizer
with open('models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model_rf, model_file)

with open('models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
