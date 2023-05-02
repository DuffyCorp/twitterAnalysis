import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import keras

import pandas as pd

import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re

train = pd.read_table('input/train.txt', delimiter=';', header=None);
val = pd.read_table('input/val.txt', delimiter=';', header=None);
test = pd.read_table('input/test.txt', delimiter=';', header=None);

data = pd.concat([train, val , test])

data.columns = ["text", "label"]

data.shape

data.isna().any(axis=1).sum()

ps = PorterStemmer()

def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line)#leave only characters from a to z
    review = review.lower() # lower the text
    review = review.split() #turn string into list of words

    #apply stemming

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # delete stop words like I, and OR
    #turn list into sentences
    return " ".join(review)

data['text']=data['text'].apply(lambda x: preprocess(x))

label_encoder = preprocessing.LabelEncoder()
data['N_label'] = label_encoder.fit_transform(data['label'])

print(data['text'])

# Creating the bag of words model

cv = CountVectorizer(max_features=5000,ngram_range=(1,3))#example: the course was long-> [the,the course,the course was,course, course was, course was long,...]

data_cv = cv.fit_transform(data['text']).toarray()

print(data_cv)

#X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(data_cv, data['N_label'], test_size=0.25, random_state=42)

#Create model
model = Sequential()
model.add(Dense(12, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=10)

_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

text='I feel sad'
text=preprocess(text)
array = cv.transform([text]).toarray()
pred = model.predict(array)

print(pred)

a=np.argmax(pred, axis=1)

result = label_encoder.inverse_transform(a)[0]
print("result",result)
print("confidence", np.max(pred, axis=1)[0])

tf.keras.models.save_model(model,'output/emotion_model.h5')

import pickle

pickle.dump(label_encoder, open('output/encoder.pkl', 'wb'))
pickle.dump(cv, open('output/CountVectorizer.pkl', 'wb'))
pickle.dump(preprocess, open('output/preprocess.pkl', 'wb'))