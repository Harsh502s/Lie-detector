import streamlit as st
import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np
import pandas as pd
import nltk 

@st.cache_data()
@st.cache_resource()
def load_model():
    model=tf.keras.models.load_model(r"lie_detector_1.h5")
    return model

loaded_model = load_model()

nltk.download('averaged_perceptron_tagger') #used for tagging words with their parts of speech (POS)
nltk.download('wordnet')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')
import spacy
nlp = spacy.load("en_core_web_sm")

def sentiment_score(text):
  # Create a Vader SentimentIntensityAnalyzer object
  analyzer = SentimentIntensityAnalyzer()
  # Get sentiment scores (compound score for overall sentiment)
  sentiment = analyzer.polarity_scores(text)
  return sentiment["compound"]

def named_entities(text):
  # Create a spaCy document
  doc = nlp(text)
  # Extract named entities and their labels (PERSON, ORG, etc.)
  entities = [(entity.text, entity.label_) for entity in doc.ents]
  return entities


def preprocess_quote(quote):
  adv_count = 0
  adj_count = 0
  noun_count = 0
  verb_count = 0
  det_count = 0
  words = quote.split()
  tagged_words = nltk.pos_tag(words)
  for word, tag in tagged_words:
    if tag.startswith('RB'):
      adv_count += 1
    elif tag.startswith('JJ'):
      adj_count += 1
    elif tag.startswith('NN'):
      noun_count += 1
    elif tag.startswith('VB'):
      verb_count += 1
    elif tag.startswith('DT'):
      det_count += 1
  word_count = len(quote.split())
  word_length = sum(len(word) for word in quote.split())
  sentiment = sentiment_score(quote)
  named_entitity = named_entities(quote)
  named_entities_count = len(named_entitity)
    
  return word_count,word_length,adv_count,adj_count,noun_count,verb_count,det_count,sentiment, named_entities_count

st.markdown("<h1 style='text-align: center;'>Lie Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter a sentence:</h3>", unsafe_allow_html=True)
sentence = st.text_input("")
if st.button("Detect Lie", use_container_width = True):
    # Call your functions and make the prediction
    processed_sentence = pd.DataFrame(preprocess_quote(sentence))
    #st.write(processed_sentence.transpose())
    #new_test = pd.DataFrame([16.0,	5.500000,	1.0,	3.0,	5.0,	1.0,	3.0,	-0.3612,	2	])
    prediction = loaded_model.predict(processed_sentence.transpose())

    new_quote_prediction_class = np.argmax(prediction)  
    #st.write(new_quote_prediction_class)  

    # Print the prediction
    if new_quote_prediction_class == 0:
      st.write("The new quote is **likely to be true**.")
    elif new_quote_prediction_class == 1:
      st.write("The new quote is **half-true**.")
    else:
      st.write("The new quote is **likely to be false or pants-on-fire**.")
