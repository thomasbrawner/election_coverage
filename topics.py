## ----------------------------------------------------- ##
## ----------------------------------------------------- ##

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns 
from nltk.corpus import stopwords
from nmf import NMFactor 
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

## ----------------------------------------------------- ##
## read the data 

data = pd.read_pickle('data/article_df.pkl')

data['source'] = np.where(data['source'] == 'The New York Times', 'New York Times', 
				 np.where(data['source'] == 'WSJ', 'Wall Street Journal', 
				 data['source']))

## ----------------------------------------------------- ##
## term-document matrix

augmented_stop_words = stopwords.words('english') + \
	['guardian','new york times','nyt','wall street journal','wsj',
	 'mr','mrs','ms','dr','gov','sen','rep',
	 'said','would','wouldn','ha','wa']

tfidf = TfidfVectorizer(stop_words = augmented_stop_words, max_features = 2000)
features = tfidf.fit_transform(data['text'])
feature_names = tfidf.get_feature_names() 

## --------------------------------------------------------------------- ##
## topics

nmf = NMFactor(k = 7)
nmf.solve(V = features, feature_names = feature_names, labels = data['source']) 

## --------------------------------------------------------------------- ##
## plot topics by source 

for i in xrange(nmf.k):
	nmf.plot_topic(i)

## --------------------------------------------------------------------- ##
## --------------------------------------------------------------------- ##
