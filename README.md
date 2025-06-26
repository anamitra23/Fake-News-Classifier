# Fake-News-Classifier
This project is an AI-based fake news detection system that uses machine learning to classify news articles as **REAL** or **FAKE** based on their content.

## 👨‍💻 Author
**Anamitra Chatterjee**

## 📌 Features
- Trained on real-world news datasets from Kaggle
- Uses TF-IDF vectorization to convert text into machine-readable format
- Passive Aggressive Classifier used for efficient training
- Predicts whether custom user input is real or fake
- Achieves around 93% accuracy

## 🧠 Tech Stack
- Python
- Scikit-learn
- Pandas
- TfidfVectorizer (NLP)
- Google Colab / Jupyter Notebook

## 📂 Dataset
The dataset used is publicly available on Kaggle:  
👉 [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

We used the `Fake.csv` and `True.csv` files to train the model.

## ▶️ How to Run

### On Google Colab (Recommended)
Click the badge below to open in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `Fake.csv` and `True.csv` to your session
2. Run the notebook cells step by step

### On Your Local Machine
```bash
pip install -r requirements.txt
python fake_news_classifier.py
```

## 📷 Sample Output
- Accuracy: ~93%
- Confusion Matrix
- Real-time prediction based on input

## 📄 License
For educational use only.
