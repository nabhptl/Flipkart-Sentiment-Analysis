import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("final_flipkart_sentiment.csv")

# Create sentiment
def sentiment(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Rating'].apply(sentiment)

# Train model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Product_name'])
y = df['Sentiment']

model = LogisticRegression()
model.fit(X, y)

# UI
st.title("📱 Product Sentiment Predictor")

product = st.text_input("Enter Product Name")

if st.button("Predict"):
    if product.strip() != "":
        vec = tfidf.transform([product])
        pred = model.predict(vec)[0]
        st.success(f"Predicted Sentiment: {pred}")
    else:
        st.warning("Please enter a product name")
        