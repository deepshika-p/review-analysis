import re
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords

def text_clean(text):
    text = text.lower()
    text = re.sub('[^a-z A-Z 0-9-]+', '', text)
    stop_words = stopwords.words('english')
    text = " ".join([i for i in text.split() if i not in stop_words or i=='not'])
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(text))
    text= " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def predict(rev):
    text_clean(rev)
    model=joblib.load('model.sav')
    tfid=joblib.load('tfidf.sav')
    rev=rev.split('\n')
    rev=tfid.transform(rev).toarray()
    return model.predict(rev)

st.title('Sentiment Analyzer')
st.markdown('ML model to scrap customer reviews from Flipkart and analyze them to determine the sentiment.')
rev=st.text_input('Review')

if st.button('Predict sentiment'):
    sentiment=predict(rev)
    if(sentiment==1):
        st.text("The review is positive")
    else:
        st.text("The review is negative")
    
