import re
import nltk
import random
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pickle


nltk.download('punkt')
nltk.download('stopwords')

# Setting as large the xtick and ytick font sizes in graphs

plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'

df = pd.read_csv('./input/IMDB Dataset.csv')
print(df)



print('\033[1m' + 'df.shape:' + '\033[0m', df.shape)
print('\033[1m' + 'df.columns:' + '\033[0m', df.columns, '\n')
print('\033[1m' + 'df.sentiment.value_counts():' + '\033[0m')
print(df.sentiment.value_counts(), '\n')

with sns.axes_style("darkgrid"):
    df['sentiment'].value_counts().plot.bar(color=['darkblue', 'r'], rot=0, fontsize='large')
    plt.savefig("./output/PositiveNegativeGraph.png")

print('\033[1m' + 'df.info:' + '\033[0m')
df.info()



df.sentiment = [1 if s == 'positive' else 0 for s in df.sentiment]
print(df)



# Storing in "before_process" a random example of review before preprocessing
# Defining and applying the function "process" performing the transformations of the reviews
# Storing in "after_process" the example of review after preprocessing

idx = random.randint(0, len(df)-1)
before_process = df.iloc[idx][0]

def process(x):
    x = re.sub('[,\.!?:()"]', '', x)
    x = re.sub('<.*?>', ' ', x)
    x = re.sub('http\S+', ' ', x)
    x = re.sub('[^a-zA-Z0-9]', ' ', x)
    x = re.sub('\s+', ' ', x)
    return x.lower().strip()

df['review'] = df['review'].apply(lambda x: process(x))
after_process = df.iloc[idx][0]



# Storing in "sw_set" the set of English stopwords provided by nltk
# Defining and applying the function "sw_remove" which remove stopwords from reviews
# Storing in "after_removal" the example of review after removal of the stopwords

sw_set = set(nltk.corpus.stopwords.words('english'))

def sw_remove(x):
    words = nltk.tokenize.word_tokenize(x)
    filtered_list = [word for word in words if word not in sw_set]
    return ' '.join(filtered_list)

df['review'] = df['review'].apply(lambda x: sw_remove(x))
after_removal = sw_remove(after_process)

def convertTuple(tup):
    str = ''.join(tup)
    return str

with open('./output/processingTextExample.txt', 'w') as f:
    P1 = 'Review #%d before preprocessing:' % idx + '\n', before_process, '\n\n'
    P2 = 'Review #%d after preprocessing:' % idx + '\n', after_process, '\n\n'
    P3 = 'Review #%d after preprocessing and stopwords removal:' % idx + '\n', after_removal
    print(P1)
    f.writelines([convertTuple(P1), convertTuple(P2), convertTuple(P3)])

# print('\033[1m' + 'Review #%d before preprocessing:' % idx + '\033[0m' + '\n', before_process, '\n')
# print('\033[1m' + 'Review #%d after preprocessing:' % idx + '\033[0m' + '\n', after_process, '\n')
# print('\033[1m' + 'Review #%d after preprocessing and stopwords removal:' % idx + '\033[0m' + '\n', after_removal)



from sklearn.model_selection import train_test_split

train_rev, test_rev, train_sent, test_sent = train_test_split(df['review'], df['sentiment'], test_size=0.1, random_state=42)

print('\033[1m' + 'train_rev.shape:' + '\033[0m', train_rev.shape)
print('\033[1m' + 'test_rev.shape:' + '\033[0m', test_rev.shape)
print('\033[1m' + 'train_sent.shape:' + '\033[0m', train_sent.shape)
print('\033[1m' + 'test_sent.shape:' + '\033[0m', test_sent.shape)



from keras.preprocessing.text import Tokenizer

dict_size = 35000
tokenizer = Tokenizer(num_words=dict_size)
tokenizer.fit_on_texts(df['review'])

print('\033[1m' + 'Dictionary size:' + '\033[0m', dict_size)
print('\033[1m' + 'Length of the tokenizer index:' + '\033[0m', len(tokenizer.word_index))
print('\033[1m' + 'Number of documents the tokenizer was trained on:' + '\033[0m', tokenizer.document_count, '\n')
print('\033[1m' + 'First 20 entries of the tokenizer index:' + '\033[0m')
print(*list(tokenizer.word_index.items())[:20])

print("train rev",train_rev)

train_rev_tokens = tokenizer.texts_to_sequences(train_rev)
test_rev_tokens = tokenizer.texts_to_sequences(test_rev)
seq_lengths =  np.array([len(sequence) for sequence in train_rev_tokens])



# Storing in "upper_bound" our chosen upper bound for the length of sequences
# Computing the percentage of lengths smaller or equal than "upper_bound"

upper_bound = int(np.mean(seq_lengths) + 2 * np.std(seq_lengths))
percentage = stats.percentileofscore(seq_lengths, upper_bound)

print('The value of upper_bound is %d and the percentage of sequences in "train_rev_tokens" \
of length smaller or equal than upper_bound is %.2f%%.' % (upper_bound, round(percentage, 2)))

# Histogram plot of the lengths of the sequences in "train_rev_tokens"

with sns.axes_style("darkgrid"):

    _, hist = plt.subplots(figsize=(10,6))
    hist.hist(seq_lengths[seq_lengths < 2*upper_bound], color='darkblue', bins=40, rwidth=0.7)
    hist.axvline(np.mean(seq_lengths), color='darkorange', linestyle='--', label='Mean value')
    hist.axvline(upper_bound, color='r', linestyle='--', label='Upper bound')

    plt.xlabel('Length of sequences in "train_rev_tokens"', size='large')
    plt.ylabel('Number of samples', size='large')
    plt.text(upper_bound, 0, 'test')
    plt.legend(fontsize='large', facecolor='palegreen')
    plt.xticks(rotation=30)
    plt.savefig("./output/HistogramOfLengthsOfSequence.png")



from keras.preprocessing.sequence import pad_sequences

train_rev_pad = pad_sequences(train_rev_tokens, maxlen=upper_bound)
test_rev_pad = pad_sequences(test_rev_tokens, maxlen=upper_bound)

print('\033[1m' + 'train_rev_pad.shape:' + '\033[0m', train_rev_pad.shape)
print('\033[1m' + 'test_rev_pad.shape:' + '\033[0m', test_rev_pad.shape, '\n')

# Printing an example of review after padding

idx_pad = random.randint(0, len(train_rev_pad)-1)
print('\033[1m' + 'Review #%d after padding:' %idx_pad + '\033[0m' + '\n', train_rev_pad[idx_pad])



from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense


output_dim = 14
units_lstm = 8
r = 0.2

model = Sequential()
model.add(Embedding(input_dim=dict_size, output_dim=output_dim, input_length=upper_bound))
model.add(Dropout(r))
model.add(LSTM(units_lstm))
model.add(Dropout(r))
model.add(Dense(1, activation='sigmoid'))

model.summary()

from keras.utils.vis_utils import plot_model

print(plot_model(model, show_shapes=True))

model.compile(optimizer='adam', loss='bce', metrics='accuracy')



validation_split = 0.1
batch_size = 384
epochs = 3

fitted = model.fit(train_rev_pad, train_sent, validation_split=validation_split,
                   batch_size=batch_size, epochs=epochs, shuffle=True)


model.save("./output/model.h5")
pickle.dump(tokenizer, open("./output/tokenizer.pkl", "wb"), protocol=0)

# Storing in "ep_values" the values of the epochs

ep_values = range(1, epochs+1)

# Plot of the training loss and validation loss (binary cross-entropy)

with sns.axes_style("darkgrid"):

    _, (loss, acc) = plt.subplots(1, 2, figsize=(15, 6))
    loss.plot(ep_values, fitted.history['loss'], color='darkblue', linestyle='dotted',
              marker='o', label='Training loss (binary cross-entropy)')
    loss.plot(ep_values, fitted.history['val_loss'], color='r', linestyle='dotted',
              marker='o', label='Validation loss (binary cross-entropy)')
    loss.set_xlabel('Epoch', size='large')
    loss.legend(fontsize='large', facecolor='palegreen')

    acc.plot(ep_values, fitted.history['accuracy'], color='darkblue', linestyle='dotted',
             marker='o', label='Training accuracy')
    acc.plot(ep_values, fitted.history['val_accuracy'], color='r', linestyle='dotted',
             marker='o', label='Validation accuracy')
    acc.set_xlabel('Epoch', size='large')
    acc.legend(fontsize='large', facecolor='palegreen')

    plt.savefig("./output/TrainValLossGraphs.png")

    result= model.evaluate(test_rev_pad, test_sent)



from sklearn.metrics import confusion_matrix

predictions = np.round(model.predict(test_rev_pad))
cf_matrix = confusion_matrix(test_sent, predictions)



# Storing in "legends" the legends of each entry of the confusion matrix
# Storing in "percentages" the percentages of each entry of the confusion matrix
# Storing in "labels" the grouped values (legend + percentage) of each entry of the confusion matrix

legends = ['True negatives', 'False positives', 'False negatives', 'True positives']
percentages = [round(100*num, 2) for num in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f'{v1}\n\n{v2}%' for v1, v2 in zip(legends, percentages)]
labels = np.asarray(labels).reshape(2, 2)

# Heatmap plot of the confusion matrix

plt.figure(figsize = (7, 7))

cm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='vlag', annot_kws={'fontsize': 'large'})
cm.set_xlabel('Predicted sentiments', size='large')
cm.set_ylabel('Actual sentiments', size='large')
cm.xaxis.set_ticklabels(['Negative', 'Positive'])
cm.yaxis.set_ticklabels(['Negative', 'Positive'])

plt.savefig("./output/ConfusionMatrix.png")



# Storing in DataFrame "df_original" the original reviews and sentiments

df_original = pd.read_csv('./input/IMDB Dataset.csv')

# Choosing randomly a review and its sentiment in the test data

idx_test = random.randint(0, len(test_sent)-1)
idx_original = test_rev.index[idx_test]
(actual_rev, actual_sent) = df_original.iloc[idx_original]

# Storing in "prediction_sent" the predicted sentiment of the chosen review
# Storing in "probability" the probability of the predicted sentiment of the chosen review

print(test_rev_pad)

prediction = model.predict(test_rev_pad)[idx_test][0]
prediction_sent = 'positive' if prediction >= 0.5 else 'negative'
probability = round(prediction if prediction >= 0.5 else 1-prediction, 2)

# Printing the original review, its predicted sentiment and probability, and original sentiment

print('\033[1m' + 'Review #%d:' % idx_original + '\033[0m' + '\n', actual_rev, '\n')
print('\033[1m' + 'Predicted sentiment:' + '\033[0m', prediction_sent, '(with probability %.2f)' % probability, '\n')
print('\033[1m' + 'Actual sentiment:' + '\033[0m', actual_sent)



def SentimentAnalysis(input):

    print(input)

    processed_text = process(input)

    # Tokenize text
    input_pad = pad_sequences(tokenizer.texts_to_sequences([processed_text]), maxlen=upper_bound)
    # Predict
    score = model.predict([input_pad])[0]

    InputPrediction_sent = 'positive' if score >= 0.5 else 'negative'

    print(score)

    print('\033[1m' + 'Predicted sentiment:' + '\033[0m', InputPrediction_sent)
    
SentimentAnalysis("I love you")

SentimentAnalysis("I hated the product, worst movie I have ever seen, So stupid")

SentimentAnalysis("I loved the movie!")