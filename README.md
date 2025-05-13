
# 🎬 IMDb Review Sentiment Analysis Web App

A web-based application that predicts the sentiment (Positive or Negative) of user-submitted IMDb movie reviews using Machine Learning (Logistic Regression + TF-IDF). The app also supports user feedback to improve model accuracy over time.

---

## 🚀 Features

- Predicts **Positive** or **Negative** sentiment for IMDb reviews.
- Uses **TF-IDF Vectorizer** and **Logistic Regression**.
- Built with **Flask** as backend.
- Allows users to submit feedback if prediction is wrong.
- Supports **model retraining** from feedback data.
- Clean, dark-themed user interface (HTML/CSS).

---

## 📁 Project Structure

```
IMDb_Review_Analysis/
│
├── app/                          # Flask backend
│   └── app.py                    # Main Flask application
│
├── data/                         # Data folder
│   ├── test.csv                  # Raw test data
│   ├── processed_test_reviews.csv # Cleaned data
│   └── feedback_data.csv         # Feedback for retraining
│
├── model/                        # Trained model and vectorizer
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html                # Frontend HTML (black theme)
│
├── preprocess.py                 # Text cleaning script
├── train_model.py                # Model training script
└── README.md                     # Project overview
```

---

## ⚙️ How to Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/IMDb_Review_Analysis.git
   cd IMDb_Review_Analysis
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run preprocessing**  
   ```bash
   python preprocess.py
   ```

4. **Train the model**  
   ```bash
   python train_model.py
   ```

5. **Start Flask app**  
   ```bash
   cd app
   python app.py
   ```

6. Open browser:  
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔁 Feedback & Retraining

- Users can submit feedback if prediction is wrong.
- Run `/retrain` route to retrain the model using feedback:
  
```bash
curl -X POST http://127.0.0.1:5000/retrain
```

---

## 📦 Requirements

- Python 3.8+
- Flask
- pandas, sklearn, nltk, joblib

Install all with:

```bash
pip install flask pandas scikit-learn nltk joblib
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributions

Feel free to fork and raise a PR to improve it!
