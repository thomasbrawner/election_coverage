
import numpy as np  
import pandas as pd 
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics 

## ----------------------------------------------------- ##
## format the data for analysis 

def format_data(data, vectorizer, testsize=0.25):
	'''
	train-test split, categorical labels, vectorized features
	'''
	features, labels = data['text'], data['source']
	x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=testsize, random_state=12345)
	x_train = vectorizer.fit_transform(x_train)
	x_test = vectorizer.transform(x_test)
	feature_names = vectorizer.get_feature_names() 
	return x_train, x_test, y_train.astype('category').values, y_test.astype('category').values

## ----------------------------------------------------- ##
## performance = accuracy (global & within class)

def accuracy(data):
	return (data['observed'] == data['predicted']).mean() 

def evaluate_classifications(classifier, data): 
	'''
	Global and within-class test accuracy. 
	'''
	x_train, x_test, y_train, y_test = data
	if isinstance(classifier, GradientBoostingClassifier):
		x_train = x_train.toarray() 
		x_test = x_test.toarray()
	classifier.fit(x_train, y_train)
	preds = classifier.predict(x_test)
	performance = pd.DataFrame([y_test, preds], columns=['observed', 'predicted'])
	class_performance = performance.groupby('observed').apply(accuracy)
	global_performance = pd.Series(accuracy(performance), index=['Global'])
	return performance, global_performance, class_performance

## ----------------------------------------------------- ##
## read the data 

data = pd.read_pickle('data/article_df.pkl')

## ----------------------------------------------------- ##
## term-document matrix

augmented_stop_words = stopwords.words('english') + \
	['guardian', 'new york times', 'nyt', 'wall street journal', 'wsj',
	 'mr', 'mrs', 'ms', 'dr', 'gov', 'sen', 'rep',
	 'said', 'would', 'wouldn', 'ha', 'wa']

tfidf = TfidfVectorizer(stop_words=augmented_stop_words, max_features=2000)
analysis_data = format_data(data, vectorizer=tfidf)

## ----------------------------------------------------- ##
## classification 

nb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1) 
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05)

perf_nb = evaluate_classifications(nb, analysis_data)
perf_rf = evaluate_classifications(rf, analysis_data)
perf_gb = evaluate_classifications(gb, analysis_data)

print '\nGlobal performance\n'
print 'naive bayes       : ', perf_nb[1]
print '\nrandom forest     : ', perf_rf[1]
print '\ngradient boosting : ', perf_gb[1]

print '\n\nWithin-class performance\n'
print 'naive bayes       : \n', perf_nb[2]
print '\nrandom forest     : \n', perf_rf[2]
print '\ngradient boosting : \n', perf_gb[2]

## ----------------------------------------------------- ##
## ----------------------------------------------------- ##
