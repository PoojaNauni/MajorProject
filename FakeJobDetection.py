import re
import string
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
df = pd.read_csv('fake_job_postings.csv')
df.head()
df.isnull().sum()
columns = ['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']
for colu in columns:
    del df[colu]
df.head()
df.fillna('',inplace=True)
plt.figure(figsize=(15,5))
sns.countplot(y='fraudulent', data=df)
plt.show()
df.groupby('fraudulent')['fraudulent'].count()
exp = dict(df.required_experience.value_counts())
del exp['']
exp
plt.figure(figsize=(10,5))
sns.set_theme(style="whitegrid")
plt.bar(exp.keys(), exp.values())
plt.title('No. of Jobs with Experience', size=20)
plt.xlabel('Experience', size=10)
plt.ylabel('No. of Jobs', size=10)
plt.xticks(rotation=30)
plt.show()
def split(location):
    l = location.split(',')
    return l[0]
df['country'] = df.location.apply(split)
df.head()
countr = dict(df.country.value_counts()[:14])
del countr['']
countr
plt.figure(figsize=(8,6))
plt.title('Country-wise Job Posting', size=20)
plt.bar(countr.keys(), countr.values())
plt.ylabel('No. of Jobs', size=10)
plt.xlabel('Countries', size=10)
[2/11, 21:31] Neha: edu = dict(df.required_education.value_counts()[:7])
del edu['']
edu
plt.figure(figsize=(15,6))
plt.title('Education-Level-wise Job Postings', size=20)
plt.bar(edu.keys(), edu.values())
plt.ylabel('No. of Jobs', size=10)
plt.xlabel('Education', size=10)
 print(df[df.fraudulent==0].title.value_counts()[:10])
 print(df[df.fraudulent==1].title.value_counts()[:10])
df['text'] = df['title']+' '+df['company_profile']+' '+df['description']+' '+df['requirements']+' '+df['benefits']
del df['title']
del df['location']
del df['department']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['required_education']
del df['required_experience']
del df['industry']
del df['function']
del df['country']
df.head()
fraudulent_text = df[df.fraudulent==1].text
realjobs_text = df[df.fraudulent==0].text
STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16,14))
wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800, stopwords=STOPWORDS).generate(str(" ".join(fraudulent_text)))
plt.imshow(wc, interpolation='bilinear')
 STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16,14))
wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800, stopwords=STOPWORDS).generate(str(" ".join(realjobs_text)))
plt.imshow(wc, interpolation='bilinear')
 punctuation = string.punctuation
 nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    
    mytokens = [word.lemma.lower().strop() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuation]
    
    return mytokens
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return[clean_text(text) for text in X]
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_params(self, deep=True):
        return{}
def clean_text(text):
    return text.strip().lower()
df['text'] = df['text'].apply(clean_text)
cv = TfidfVectorizer(max_features=100)
x = cv.fit_transform(df['text'])
df1 = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())
df.drop(["text"], axis=1, inplace=True)
main_df = pd.concat([df1, df], axis=1)
main_df.head()
 Y = main_df.iloc[:,-1]
X = main_df.iloc[:,:-1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=3, oob_score=True, n_estimators=100, criterion="entropy")
model = rfc.fit(x_train, y_train)
print(x_test)
 pred = rfc.predict(x_test)
score = accuracy_score(y_test, pred)
score
print("Classification Report\n")
print(classification_report(y_test, pred))
print("Confusion Matrix\n")
print(confusion_matrix(y_test, pred))
