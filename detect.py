import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# LOAD DATASETS
# -----------------------------

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# -----------------------------
# ADD LABELS
# -----------------------------

true_df["label"] = 1     # TRUE news
fake_df["label"] = 0     # FAKE news

# -----------------------------
# COMBINE DATASETS
# -----------------------------

df = pd.concat([true_df, fake_df], axis=0)

# -----------------------------
# CREATE SINGLE CONTENT COLUMN
# -----------------------------

df["content"] = df["title"] + " " + df["text"]

# -----------------------------
# SELECT INPUT AND OUTPUT
# -----------------------------

X = df["content"]
y = df["label"]

# -----------------------------
# SPLIT DATA
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TEXT VECTORIZATION
# -----------------------------

vectorizer = TfidfVectorizer(stop_words='english')

Xv_train = vectorizer.fit_transform(X_train)
Xv_test = vectorizer.transform(X_test)

# -----------------------------
# TRAIN MODEL
# -----------------------------

model = LogisticRegression()

model.fit(Xv_train, y_train)

# -----------------------------
# TEST ACCURACY
# -----------------------------

y_pred = model.predict(Xv_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# SAVE MODEL
# -----------------------------

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and Vectorizer saved successfully!")

# -----------------------------
# TEST WITH CUSTOM NEWS
# -----------------------------

while True:

    news = input("\nEnter News Text (or type exit): ")

    if news.lower() == "exit":
        break

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        print("TRUE NEWS")
    else:
        print("FAKE NEWS")