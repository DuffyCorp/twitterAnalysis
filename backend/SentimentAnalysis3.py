from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import time
import pickle

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"

model = keras.models.load_model('./sentimentAnalysis3/output/SentimentModel.h5')

f = open('./sentimentAnalysis3/output/tokenizer.pkl', 'rb')
tokenizer = pickle.load(f)
f.close()

SENTIMENT_THRESHOLDS = (0.4, 0.7)

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def process(x):
    x = re.sub('[,\.!?:()"]', '', x)
    x = re.sub('<.*?>', ' ', x)
    x = re.sub('http\S+', ' ', x)
    x = re.sub('[^a-zA-Z0-9]', ' ', x)
    x = re.sub('\s+', ' ', x)
    return x.lower().strip()

def SentimentAnalysis(input):
    start_at = time.time()
    
    processed_text = process(input)

    # Tokenize text
    input_pad = pad_sequences(tokenizer.texts_to_sequences([processed_text]), maxlen=291)
    # Predict
    score = model.predict([input_pad])[0]

    label = decode_sentiment(score, include_neutral=True)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  
    