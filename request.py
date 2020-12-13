# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:43:23 2020

@author: majed.aljefri
"""
import requests
import json

url = 'http://127.0.0.1:5000/predict_api/'

data = 'We had a great experience at this hotel, hightly recommended'
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print('review: ',data)
print(r.text)

