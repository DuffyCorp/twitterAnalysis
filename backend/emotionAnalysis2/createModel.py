import os

import tensorflow as tf

import pickle

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import pandas as pd

import numpy as np

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import re

# DATASET
DATASET_COLUMNS = ["ids", "text", "label"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 20
BATCH_SIZE = 128

# EXPORT
KERAS_MODEL = "output/model.h5"
WORD2VEC_MODEL = "output/model.w2v"
TOKENIZER_MODEL = "output/tokenizer.pkl"
ENCODER_MODEL = "output/encoder.pkl"

def printAccuracyGraph(epochs, acc, val_acc):
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.savefig('output/AccuracyGraph.png')
    plt.clf()

def printLossGraph(epochs, loss, val_loss):
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('output/LossGraph.png')
    plt.clf()

df = pd.read_csv("input/test.csv", encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

print("Dataset size:", len(df))

df.head(5)

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

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

df.text = df.text.apply(lambda x: preprocess(x))

df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))

documents = [_text.split() for _text in df_train.text] 

# w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=8)
# w2v_model.build_vocab(documents)

# words = w2v_model.wv.key_to_index.keys()
# vocab_size = len(words)
# print("Vocab size", vocab_size)

# w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

w2v_model = gensim.models.Word2Vec.load("input/model.w2v")

print(w2v_model.wv.most_similar("love"))

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(df_train.text)

tokenizer = pickle.load(open('input/tokenizer.pkl', 'rb'))

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)

labels = df_train.label.unique().tolist()
labels

encoder = LabelEncoder()
encoder.fit(df_train.label.tolist())

y_train = encoder.transform(df_train.label.tolist())
y_test = encoder.transform(df_test.label.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=2)]

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])

def emotionAnalysis(text):
    text=preprocess(text)
    array = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    pred = model.predict(array)

    print(pred)

    a=np.argmax(pred, axis=1)

    result = encoder.inverse_transform(a)[0]
    print("result",result)
    print("confidence", np.max(pred, axis=1)[0])


emotionAnalysis("I feel sad")
emotionAnalysis("I love you")
emotionAnalysis("God I hate star wars")

if not os.path.exists("./output"):
    os.makedirs("./output")

print("Classification Report")

Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)

report = classification_report(y_test, y_pred, output_dict=True)

df = pd.DataFrame(report).transpose()
df.to_csv("output/ClassificationResults.csv", encoding='utf-8', index=False)

model.save(KERAS_MODEL)
w2v_model.save(WORD2VEC_MODEL)
pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)
pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)

with open('output/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

with open('output/trainHistoryDict', 'rb') as handle:
    history = pickle.load(handle)

if not os.path.exists('output/AccuracyGraph.png'):
    print("Accuracy graph")

    acc = history['accuracy']
    val_acc = history['val_accuracy']

    epochs = range(len(acc))

    printAccuracyGraph(epochs, acc, val_acc)

if not os.path.exists('output/LossGraph.png'):
    print("Loss graph")

    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(loss))

    printLossGraph(epochs, loss, val_loss)