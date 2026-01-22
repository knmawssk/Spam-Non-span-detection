import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords', quiet=True)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


import warnings
warnings.filterwarnings('ignore')

#importing
data = pd.read_csv("spam_ham_dataset.csv")

#just graph
sns.countplot(x='label', data=data)
plt.show()

#assigning new values
ham_msg=data[data['label']=='ham']
spam_msg=data[data['label']=='spam']
#making sure n of spam=n of ham
balanced_ham=ham_msg.sample(n=len(spam_msg), random_state=42)
balanced_data=pd.concat([balanced_ham, spam_msg]).reset_index(drop=True)

#making graph
sns.countplot(x='label', data=balanced_data)
plt.title("Spam and Ham")
plt.xticks(ticks=[0,1], labels=['Ham', 'Spam'])
plt.show()

#deleting punctuation
balanced_data['text'] = balanced_data['text'].str.replace('Subject', ' ')
print('no subject', balanced_data['text'].head())

punctuation_list = string.punctuation
def remove_punc(text):
    temp=str.maketrans('', '', punctuation_list)
    return text.translate(temp)

balanced_data['text']=balanced_data['text'].apply(lambda x: remove_punc(x))
print('no punctuation', balanced_data['text'])

#adding important words, avoiding stop words
def imp_words(text):
    stop_words = stopwords.words('english')
    imp_words = []

    for word in str(text).split():
        word = word.lower()
        if word not in stop_words:
            imp_words.append(word)
    output = " ".join(imp_words)
    return output

balanced_data['text']=balanced_data['text'].apply(lambda x: imp_words(x))
print('important words only', balanced_data['label'],  balanced_data['text'])
balanced_data.to_csv('new_file', index = False)

#showing word with visualization
def wordshow(data, typ):
    text=" ".join(data['text'])
    wc=WordCloud(background_color='black', max_words = 100, width=800, height=400).generate(text)
    plt.figure(figsize=(7,7))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'Word Count for {typ} of messages', fontsize=15)
    plt.axis='off'
    plt.show()

wordshow(balanced_data[balanced_data['label']=='ham'], typ='Non-Spam')
wordshow(balanced_data[balanced_data['label']=='spam'], typ='Spam')

train_X, test_X, train_Y, test_Y = train_test_split(
    balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)

max_len = 100  # Maximum sequence length
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

train_Y = (train_Y == 'spam').astype(int)
test_Y = (test_Y == 'spam').astype(int)

#modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lf=ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

history=model.fit(
    train_sequences, train_Y,
    validation_data=(test_sequences, test_Y),
    epochs=20,
    batch_size=32,
    callbacks=[lf, es]
)

test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('test loss:', test_loss)
print('test accuracy:', test_accuracy)

plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title='Accuracy showcase'
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()