import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Define the file path for the cleaned dataset
data_path = 'A:\\IMDb_Review_Analysis\\data\\processed_test_reviews.csv'

# Load the cleaned dataset
df = pd.read_csv(data_path)

# Drop rows with missing values just in case
df.dropna(subset=['cleaned_text', 'sentiment'], inplace=True)

# Split data into features (X) and labels (y)
X = df['cleaned_text']
y = df['sentiment']

# Split the data into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize TF-IDF Vectorizer (with max features and bigrams)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Fit and transform the training data using the vectorizer
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict and evaluate the model on the test data
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# Output the accuracy and classification report
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Define the output directory to save the model and vectorizer
output_dir = 'A:\\IMDb_Review_Analysis\\model'
os.makedirs(output_dir, exist_ok=True)

# Save the trained model using pickle
with open(os.path.join(output_dir, 'sentiment_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer using pickle
with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

# Confirmation of saved model and vectorizer
print(f"Model and vectorizer saved to: {output_dir}")
