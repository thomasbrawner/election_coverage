## ----------------------------------------------------- ##
## ----------------------------------------------------- ##

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd  
import seaborn as sns 
from scipy.sparse.linalg import lsqr 
from wordcloud import WordCloud

## ----------------------------------------------------- ##
## n largest items

def n_largest(arr, n): return np.argpartition(arr, -n)[-n:]

## ----------------------------------------------------- ##
## nmf class

class NMFactor(object):

	def __init__(self, k, max_iter = 200):
		'''
		V : term document matrix 
		k : latent topics 
		max_iter : iterations 
		feature_names : optional, names corresponding to features in V
		'''
		self.max_iter = max_iter
		self.k = k
		self.topics = ['Topic ' + str(i) for i in np.arange(1, self.k + 1)]

	def solve(self, V, labels = None, feature_names = None, converge_target = 0.0001):
		'''
		alternating least squares solution for decomposing term-doc matrix V. 
		stop updating when improvement less than converge_target or at 
		max iterations 
		'''
		self.V = V
		self.W = np.random.rand(V.shape[0], self.k)
		self.H = np.random.rand(self.k, V.shape[1])
		self.labels = labels 
		self.feature_names = feature_names
		self.rss = [np.sum(np.square( V - self.W.dot(self.H) ))]
		self.mse = None 
		self.iters = 0  

		for i in xrange(self.max_iter):

			W_temp = np.linalg.lstsq(a = self.H.T, b = self.V.todense().T )[0].T
			self.W = np.clip(W_temp, a_min = 0, a_max = W_temp.max() )
			
			H_temp = np.linalg.lstsq(a = self.W, b = self.V.todense() )[0]
			self.H = np.clip(H_temp, a_min = 0, a_max = H_temp.max() )
			
			errors = self.V - self.W.dot(self.H)
			self.rss.append(np.sum(np.square(errors)))

			if np.absolute(self.rss[-2] - self.rss[-1]) <= converge_target:
				break
			
		self.iters = i + 1
		self.mse = np.sqrt(self.rss[-1] / float(self.V.size))

	def top_words(self, topic, n_words):
		'''
		return the top n words for each of k topics 
		'''
		if not hasattr(self, 'H'):
			raise Exception('solve method needs to be executed.')
		
		word_idx = n_largest(np.array(self.H)[topic, :].flatten(), n_words)
		words = np.array(self.feature_names)[word_idx]
		values = np.array(self.H)[topic, word_idx]
		values = np.round(values * 10)
		return zip(words, values)

	def top_docs(self, n): 
		'''
		return the top n documents for each of k topics 
		'''
		if not hasattr(self, 'W'): 
			raise Exception('solve method needs to be executed.')
			
		idx_docs = []
		for i in xrange(self.W.shape[1]):
			idx_docs.append(n_largest(np.array(self.W[:, i]).flatten(), n))
		return dict(zip(np.arange(self.k), idx_docs))

	def word_string(self, topic, n_words): 
		'''
		display word cloud for topic using n_words top words for that topic. 
		the word cloud function expects a string...
		''' 
		top_words = self.top_words(topic, n_words)
		word_string = ' '.join([' '.join([x] * y) for x, y in top_words])
		return word_string 

	def topic_strength_by_label(self, topic):
		'''
		topic strength faceting on provided labels. 
		'''
		if not hasattr(self, 'W'):
			raise Exception('solve method needs to be executed.')

		if type(self.labels) is not pd.core.series.Series: 
			labels = pd.Series(self.labels)

		data = pd.concat([self.labels, pd.DataFrame(self.W)], axis = 1)
		data.columns = ['Label'] + self.topics 
		gdata = data.groupby('Label').mean()
		gdata = gdata.iloc[:,topic].reset_index() 
		gdata.columns = ['Label','Value']
		return gdata 

	def plot_topic(self, topic, n_words = 50, file_prefix = ''):
		'''
		Plot average contribution to topic across sources alongside
		word cloud (containing n_words) for that topic 
		'''
		if topic not in np.arange(self.k):
			raise Exception('topic must be an integer in the range [0, self.k - 1].')

		word_string = self.word_string(topic, n_words)
		wordcloud = WordCloud().generate(word_string)
		topic_strength = self.topic_strength_by_label(topic)

		plt.figure(figsize = (9, 3.5))

		plt.subplot(121)
		sns.barplot(x = 'Label', y = 'Value', data = topic_strength)
		plt.xlabel(''); plt.ylabel('')

		plt.subplot(122)
		plt.imshow(wordcloud)
		plt.axis('off')

		plt.tight_layout() 
		plt.savefig('figures/' + file_prefix + 'source_topic' + str(topic) + '.png') 
		plt.close() 
		return 
			
## ----------------------------------------------------- ##
## ----------------------------------------------------- ##