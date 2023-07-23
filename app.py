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
    text = str(text).lower()
    text = re.sub('[^a-z A-Z 0-9]+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    stop_words = stopwords.words('english')
    text = " ".join([i for i in text.split() if i not in stop_words or i=='not'])
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(text))
    text= " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def predict(rev):
    rev=text_clean(rev)
    model=joblib.load('model.sav')
    tfid=joblib.load('tfidf.sav')
    rev=rev.split(' ')
    rev=tfid.transform(rev).toarray()
    return model.predict(rev)

st.title('Sentiment Analyzer')
st.markdown('ML model to scrap customer reviews from Flipkart and analyze them to determine the sentiment.')

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

#page_bg_img = '''
#<style>
#.stApp {
#background-image: url("https://imageio.forbes.com/specials-images/imageserve/612a5fe11444398a55b0da5e/AI-enabled-sentiment-analysis/960x0.jpg?format=jpg&width=960");
#background-size: cover;
#}
#</style>
#''' 
#st.markdown(page_bg_img, unsafe_allow_html=True)

rev=st.text_input('Review')
if st.button('Predict sentiment'):
    sentiment=predict(rev)
    if(sentiment.all()==1):
        st.success('The review is positive.')
        #st.markdown(f'<p style="font-family: Arial, Helvetica, sans-serif;color:#2dc937;font-size:24px;">{"The review is positive"}</p>', unsafe_allow_html=True)
    else:
        st.error("The review is negative.")
        #st.markdown(f'<p style="font-family: Arial, Helvetica, sans-serif;color:#cc3232;font-size:24px;">{"The review is negative"}</p>', unsafe_allow_html=True)
    
