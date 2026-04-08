import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = 0
true["label"] = 1
df = pd.concat([fake, true])
df = df[["text", "label"]]
df = df.dropna()
df = df.sample(5000)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2
)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
def predict_news(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = max(prob)
    return result, confidence
st.title("📰VeriNews- AI Fake News Detector")

st.write("This app uses Machine Learning (NLP) to detect fake news.")
st.info("Model: TF-IDF + Logistic Regression")

st.write(f"Model Accuracy: {accuracy:.2f}")

user_input = st.text_area("Enter news text:")

if st.button("Analyze"):
    if user_input:
        result, confidence = predict_news(user_input)

        if result == 1:
            st.success("REAL NEWS ✅")
        else:
            st.error("FAKE NEWS ❌")

        st.write(f"Confidence Score: {confidence:.2f}")