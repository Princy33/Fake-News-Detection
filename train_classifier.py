import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Load CSV Files

fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)


# Handle Missing Values (IMPORTANT FIX)

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Combine title + text
df["combined"] = df["title"] + " " + df["text"]

# Remove empty rows (if any)
df = df[df["combined"].str.strip() != ""]

X = df["combined"]
y = df["label"]


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Model

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# Evaluation

print(classification_report(y_test, model.predict(X_test_vec)))


# Save Files

joblib.dump(vectorizer, "vectorizer.jb")
joblib.dump(model, "lr_model.jb")

print("Model training complete ")
