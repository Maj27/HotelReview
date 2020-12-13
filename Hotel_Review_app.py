# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:11:00 2020

@author: majed.aljefri
"""
#app.py
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

model_file = 'C:/Side Projects/Chata ai/model.pickle'


app = Flask(__name__)
dt_model, lr_model, vec = p.load(open(model_file, 'rb'))


@app.route('/')
#def home():
#    return render_template('index.html')

@app.route('/predict_api/', methods=['POST'])
def predict_api():
    new_review = request.get_json(force = True)
    prediction = new_prediction(new_review, lr_model, vec)
    
    return jsonify(prediction)

def preprocess_text(row):
    tokens = nltk.word_tokenize(row['review'])
    
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    words = [w for w in tokens if w not in string.punctuation]
    
    # remove remaining tokens that are not alphabetic
    words = [w for w in words if w.isalpha()]
    
    # filter out stop words
    words = [w for w in words if not w in stop_words]
   
    #stems = [porter.stem(word) for word in words]
    lemmas =[lemmatizer.lemmatize(word) for word in words]
                   
    #return ' '.join(stems)
    return ' '.join(lemmas)

def new_prediction(review,model, vec):
   
    lemmas = np.array(str(preprocess_text(review)))
    features = vec.transform(lemmas.ravel())
   
    return model.predict(features)
    
    
if __name__ == '__main__':
    app.run(debug=True)