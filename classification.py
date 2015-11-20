
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

rf_grid = {'n_estimators' : [1000],
		   'max_features' : ['sqrt', 0.5], 
		   'min_samples_leaf' : [3, 7], 
		   'max_depth' : [7, None]} 
rf = RandomForestClassifier(n_jobs=-1) 
rf_search = GridSearchCV(rf, rf_grid, cv=5, n_jobs=-1, verbose=1)
rf_search.fit(analysis_data[0], np.array(analysis_data[2]))

gb_grid = {'n_estimators' : [1000],
		   'max_features' : [0.5, None],
		   'max_depth' : [1, 3, 5],
		   'learning_rate' : [0.1, 0.05, 0.01]}
gb = GradientBoostingClassifier()
gb_search = GridSearchCV(gb, gb_grid, cv=5, verbose=1)
gb_search.fit(analysis_data[0], np.array(analysis_data[2]))

## ----------------------------------------------------- ##
## performance 

perf_nb = evaluate_classifications(nb, analysis_data)
perf_rf = evaluate_classifications(rf_search.best_estimator_, analysis_data)
perf_gb = evaluate_classifications(gb_search.best_estimator_, analysis_data)

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
