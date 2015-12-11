import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd  
import seaborn as sns 
from scipy.sparse.linalg import lsqr 
from sklearn.cluster import AgglomerativeClustering, KMeans 
from wordcloud import WordCloud
sns.set_style('whitegrid')


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

    def solve(self, V, labels = None, feature_names = None, dates = None, converge_target = 0.0001): 
        '''
        alternating least squares solution for decomposing term-doc matrix V. 
        stop updating when improvement less than converge_target or at 
        max iterations. 
        return predicted topics in self.topics 
        '''
        self.V = V
        self.W = np.random.rand(V.shape[0], self.k)
        self.H = np.random.rand(self.k, V.shape[1])
        self.labels = labels 
        if type(self.labels) is not pd.core.series.Series: 
            self.labels = pd.Series(self.labels)
        self.feature_names = feature_names
        self.dates = dates
        if type(self.dates) is not pd.core.series.Series: 
            self.dates = pd.Series(self.dates)
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
        self.topic_preds = np.argmax(np.array(self.W), 1).flatten() 

    def top_words(self, topic, n_terms):
		# return the top n words for each of k topics 
        if not hasattr(self, 'H'):
            raise Exception('solve method needs to be executed.')
		
        word_idx = n_largest(np.array(self.H)[topic, :].flatten(), n_terms)
        words = np.array(self.feature_names)[word_idx]
        values = np.array(self.H)[topic, word_idx]
        values = np.round(values * 10)
        return zip(words, values)

    def top_docs(self, n): 
		# return the top n documents for each of k topics 
        if not hasattr(self, 'W'): 
            raise Exception('solve method needs to be executed.')
			
        idx_docs = []
        for i in xrange(self.W.shape[1]):
            idx_docs.append(n_largest(np.array(self.W[:, i]).flatten(), n))
        return dict(zip(np.arange(self.k), idx_docs))

    def word_string(self, topic, n_terms): 
        '''
        display word cloud for topic using n_terms top words for that topic. 
        the word cloud function expects a string.
        ''' 
        top_words = self.top_words(topic, n_terms)
        word_string = ' '.join([' '.join([x] * y) for x, y in top_words])
        return word_string 

    def topic_strength_by_label(self, topic):
        # topic strength faceting on provided labels. 
        if not hasattr(self, 'W'):
            raise Exception('solve method needs to be executed.')

        data = pd.concat([self.labels, pd.DataFrame(self.W)], axis = 1)
        data.columns = ['Label'] + self.topics 
        gdata = data.groupby('Label').mean()
        gdata = gdata.iloc[:,topic].reset_index() 
        gdata.columns = ['Label','Value']
        return gdata 

    def topic_strength_by_label_time(self, topic):
        # topic strength faceting on label, month by month
        if not hasattr(self, 'W'):
            raise Exception('solve method needs to be executed.')

        months = pd.Series(pd.DatetimeIndex(self.dates).month)
        data = pd.concat([self.labels, months, pd.DataFrame(self.W)], axis = 1)
        data.columns = ['Label','Month'] + self.topics
        data = data.query('Month < 9')
        gdata = data.groupby(['Label','Month']).mean()
        gdata = gdata.iloc[:,topic].reset_index()  
        gdata.columns = ['Label','Month','Value']
        return gdata 

    def plot_topic(self, topic, n_terms = 50, file_prefix = ''):
        '''
        Plot average contribution to topic across sources alongside
        word cloud (containing n_terms) for that topic 
        '''
        if topic not in np.arange(self.k):
            raise Exception('topic must be an integer in the range [0, self.k - 1].')

        word_string = self.word_string(topic, n_terms)
        wordcloud = WordCloud(background_color = 'white', width = 2200, height = 1100).generate(word_string)
        topic_strength = self.topic_strength_by_label(topic)

        monthly = self.topic_strength_by_label_time(topic)
        guardian = monthly.query('Label == "Guardian"')
        nyt = monthly.query('Label == "New York Times"')
        wsj = monthly.query('Label == "Wall Street Journal"')
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']

        plt.figure(figsize = (8, 8))
        ax1 = plt.subplot2grid((5, 7), (0, 0), rowspan = 3, colspan = 7)
        ax2 = plt.subplot2grid((5, 7), (3, 0), rowspan = 2, colspan = 3)
        ax3 = plt.subplot2grid((5, 7), (3, 3), rowspan = 2, colspan = 4)

        ax1.imshow(wordcloud)
        ax1.axis('off')

        ax2.plot(guardian['Month'], guardian['Value'], color = 'b', label = 'Guardian')
        ax2.plot(nyt['Month'], nyt['Value'], color = 'g', label = 'New York Times')
        ax2.plot(wsj['Month'], wsj['Value'], color = 'r', label = 'Wall Street Journal')
        plt.xlabel(''); plt.ylabel(''); ax2.set_xticklabels(months)

        sns.barplot(x = 'Label', y = 'Value', data = topic_strength)
        plt.xlabel(''); plt.ylabel('')

        plt.tight_layout() 
        plt.savefig('figures/' + file_prefix + 'source_topic' + str(topic) + '.png') 
        plt.close() 
        return 



def n_largest(arr, n): return np.argpartition(arr, -n)[-n:]

def scale_conversion(df, new_scale):
    '''
    convert values of df from arbitrary range to (min, max), then 
    round the result to integer
    '''
    old_range = (df.max().max() - df.min().min())  
    new_range = (new_scale[1] - new_scale[0])  
    new_df = (((df - df.min().min()) * new_range) / old_range) + 1
    return np.round(new_df)
