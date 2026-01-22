import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

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
print('no subject', balanced_data['text'].head)

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

