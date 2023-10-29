import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load your dataset
# Replace 'your_data.csv' with your dataset file.
data = pd.read_csv('data.csv')

# Data preprocessing
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['label']

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("Hindi SMS Spam Detection")

user_input = st.text_input("Enter an SMS message in Hindi:")
if st.button("Detect"):
    input_data = tfidf_vectorizer.transform([user_input])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Spam SMS")
    else:
        st.success("Not a Spam SMS")

# st.write("Disclaimer: This is a simplified example. For a more accurate model, consider using a larger dataset and more advanced techniques.")

# # Run the Streamlit app
# if __name__ == '__main__':
#     st.run()
