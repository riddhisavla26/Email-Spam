import streamlit as st
import pickle
import string
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.markdown(
    '<p class="main-title">🛡 SpamShield AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Advanced Email & SMS Spam Detection System</p>',
    unsafe_allow_html=True
)

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown(
        '<div class="spam">🚨 SPAM MESSAGE DETECTED</div>',
        unsafe_allow_html=True
    )
    else:
        st.markdown(
        '<div class="ham">✅ SAFE / LEGITIMATE MESSAGE</div>',
        unsafe_allow_html=True
    )