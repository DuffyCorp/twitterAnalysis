import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import time


ps = PorterStemmer()

encoder = pickle.load(open('emotionAnalysis/output/encoder.pkl', 'rb'))

cv = pickle.load(open('emotionAnalysis/output/CountVectorizer.pkl', 'rb'))

model=tf.keras.models.load_model('emotionAnalysis/output/emotion_model.h5')

def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line)#leave only characters from a to z
    review = review.lower() # lower the text
    review = review.split() #turn string into list of words

    #apply stemming

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # delete stop words like I, and OR
    #turn list into sentences
    return " ".join(review)

def emotionAnalysis(text):
    start_at = time.time()

    text =preprocess(text)

    # Convert to array
    array = cv.transform([text]).toarray()
    # Predict
    pred = model.predict(array)
    # Decode sentiment
    a=np.argmax(pred, axis=1)
    prediction = encoder.inverse_transform(a)[0]

    score = np.max(pred, axis=1)[0]

    return {"label": prediction, "score": float(score),
       "elapsed_time": time.time()-start_at}  

print(emotionAnalysis("I feel sad"))
print(emotionAnalysis("God I hate star wars"))
print(emotionAnalysis("Im happy"))
print(emotionAnalysis("I like men"))