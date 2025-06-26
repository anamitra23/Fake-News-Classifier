import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the datasets
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = 0  # FAKE
real_df["label"] = 1  # REAL

# Combine and shuffle
data = pd.concat([fake_df, real_df], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# Features and labels
X = data['text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model: Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Sample prediction
test_news = """
NASA's Artemis I mission successfully launched today from Kennedy Space Center. 
This uncrewed mission is the first in a series of increasingly complex missions to build a 
long-term human presence on the Moon. Scientists worldwide are praising the accomplishment.
"""
vec = tfidf.transform([test_news])
result = model.predict(vec)
print("Prediction:", "REAL ✅" if result[0] == 1 else "FAKE ❌")
