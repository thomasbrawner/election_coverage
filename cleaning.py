## ----------------------------------------------------- ##
## ----------------------------------------------------- ##

import numpy as np  
import pandas as pd 
import re, string 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from pymongo import MongoClient

## ----------------------------------------------------- ##
## update - rename timestamps in mongo shell

"""
> db.articles.update({'source' : 'WSJ'}, {$rename : {'timestamp' : 'article_date'}})
WriteResult({ "nMatched" : 1, "nUpserted" : 0, "nModified" : 1 })
> db.articles.update({'source' : 'WSJ'}, {$rename : {'timestamp' : 'article_date'}}, {multi : true})
WriteResult({ "nMatched" : 1282, "nUpserted" : 0, "nModified" : 1281 })
> db.articles.update({'source' : 'The New York Times'}, {$rename : {'pub_date' : 'article_date'}}, {multi : true})
WriteResult({ "nMatched" : 4190, "nUpserted" : 0, "nModified" : 4190 })
> db.articles.update({'source' : 'Guardian'}, {$rename : {'webPublicationDate' : 'article_date'}}, {multi : true})
WriteResult({ "nMatched" : 3271, "nUpserted" : 0, "nModified" : 3271 })
"""

## ----------------------------------------------------- ##
## candidates

candidates = '|'.join([
	'lincoln chafee',
	'hillary clinton','hillary rodham clinton',
	"martin o'malley",
	'bernie sanders',
	'jim webb',
	'elizabeth warren',
	'joe biden','joseph biden',
	'jeb bush',
	'ben carson',
	'chris christie',
	'ted cruz',
	'carly fiorina',
	'jim gilmore',
	'lindsey graham',
	'mike huckabee',
	'bobby jindal',
	'john kasich',
	'george pataki',
	'rand paul',
	'rick perry',
	'marco rubio',
	'rick santorum',		 
	'donald trump',
	'scott walker',
	'mitt romney',
	])

## ----------------------------------------------------- ##
## programs to clean text

def remove_emails(doc): 
	p = re.compile(r'[\w]+[@][\w\.]+')
	iterator = p.finditer(doc)
	for match in iterator:
		target = doc[match.span()[0]:match.span()[1]]
		doc = string.replace(doc, target, '')
	return doc 

def clean_dates(date_string):
	return pd.to_datetime(date_string.replace('Updated', '').strip())

def clean_source_text(articles, filter_terms):
	'''
	Filter articles by relevance to candidates, also 
	remove email addresses (e.g., author@wsj.com) from the text
	'''
	data = pd.DataFrame([article for article in articles.find()])
	data['new_date'] = data['article_date'].apply(lambda x: clean_dates(x))
	data['date'] = pd.DatetimeIndex(data['new_date']).date 
	data['month'] = pd.DatetimeIndex(data['new_date']).month
	data['text'] = data['text'].str.lower() 
	data = data[data['text'].str.contains(filter_terms).fillna(False)]
	data['text'] = data['text'].apply(lambda x: remove_emails(x))
	return data[['source','date','month','text']]

def clean_wsj(data):
	'''
	Trim leading article metadata for WSJ
	'''
	wsj = data.query('source == "WSJ"')['text'].tolist()
	labels = data.query('source == "WSJ"')['source'].tolist() 
	date = data.query('source == "WSJ"')['date'].tolist()
	month = data.query('source == "WSJ"')['month'].tolist() 
	data = data.query('source != "WSJ"')
	wsj = [x[(x.find('comments') + len('comments')):] if x.find('comments') >= 0 else x for x in wsj]
	wsj = pd.DataFrame(zip(labels, wsj, date, month))
	wsj.columns = ['source','text', 'date', 'month']
	return pd.concat([data, wsj]) 

def stem_data(data): 
	'''
	Lemmatization
	'''
	features = data['text'].tolist() 
	features = [' '.join([WordNetLemmatizer().lemmatize(x) for x in doc.split()]) for doc in features]
	data['text'] = features
	return data

## ----------------------------------------------------- ##
## read articles from mongo 

client = MongoClient(port = 27017)
db = client['election2015']
articles = db['articles']

## ----------------------------------------------------- ##
## data processing 

data = clean_source_text(articles, candidates)
data = clean_wsj(data)
data = stem_data(data)

## ----------------------------------------------------- ##
## write the resulting data frame to disk 

data.reset_index(drop = True).to_pickle('data/article_df.pkl')
client.close() 

## ----------------------------------------------------- ##
## ----------------------------------------------------- ##
