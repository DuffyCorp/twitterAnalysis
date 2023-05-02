import time
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import re
import numpy as np

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# KERAS
SEQUENCE_LENGTH = 300

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

model = keras.models.load_model('output1/model.h5')

f = open('output1/tokenizer.pkl', 'rb')
tokenizer = pickle.load(f)
f.close()

f = open('output1/encoder.pkl', 'rb')
encoder = pickle.load(f)
f.close()

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def emotionAnalysis(text):
    text=preprocess(text)
    array = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    pred = model.predict(array)

    #print(pred)

    a=np.argmax(pred, axis=1)

    result = encoder.inverse_transform(a)[0]
    print("result",result)
    print("confidence", np.max(pred, axis=1)[0])

emotionAnalysis("I feel sad")
emotionAnalysis("God I hate star wars")
emotionAnalysis("Im happy")
emotionAnalysis("I love it when she bends over like that")
