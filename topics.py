import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns 
from collections import defaultdict
from nltk.corpus import stopwords
from nmf import NMFactor 
from pattern.en import parse, split, wordnet
from random import seed  
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
seed(1537)


data = pd.read_pickle('data/article_df.pkl')
data['source'] = np.where(data['source'] == 'The New York Times', 'New York Times', 
                 np.where(data['source'] == 'WSJ', 'Wall Street Journal', data['source']))

## term-document matrix
augmented_stop_words = stopwords.words('english') + \
    ['guardian', 'new york times', 'nyt', 'wall street journal', 'wsj', 'mr', 'mrs', 
     'ms', 'dr', 'gov', 'sen', 'rep', 'said', 'would', 'wouldn', 'ha', 'wa']

tfidf = TfidfVectorizer(stop_words = augmented_stop_words, max_features = 2000)
features = tfidf.fit_transform(data['text'])
feature_names = tfidf.get_feature_names() 

## topics
nmf = NMFactor(k = 15)
nmf.solve(V = features, feature_names = feature_names, dates = data['date'], labels = data['source']) 
data['topic'] = nmf.topic_preds 

## within topics 
def scale_conversion(df, new_scale):
    '''
    convert values of df from arbitrary range to (min, max), then 
    round the result to integer
    '''
    old_range = (df.max().max() - df.min().min())  
    new_range = (new_scale[1] - new_scale[0])  
    new_df = (((df - df.min().min()) * new_range) / old_range) + 1
    return np.round(new_df)

def n_largest(arr, n): return np.argpartition(arr, -n)[-n:]

def cloud_word_string(tfidf_mat, tfidf_names, count_mat, count_names):
    tfidf_data = {}
    for source in tfidf_mat.index.values:
        row = tfidf_mat.ix[source]
        term_idx = n_largest(row, 25)
        terms = np.array(tfidf_names)[term_idx]
        values = np.array(row)[term_idx]
        tfidf_data[source] = ' '.join([' '.join([x] * y) for x, y in zip(terms, values)])

    count_data = {}
    for source in count_mat.index.values:
        row = count_mat.ix[source]
        term_idx = n_largest(row, 25)
        terms = np.array(count_names)[term_idx]
        values = np.array(row)[term_idx]
        count_data[source] = ' '.join([' '.join([x] * y) for x, y in zip(terms, values)])

    return {key : tfidf_data[key] + count_data[key] for key in tfidf_data}

def within_label_cloud_by_topic(data_dict, topic):
    # Explore the topics within the labels. 
    try:
        guardian = WordCloud(background_color = 'white', width = 800, height = 1800).generate(data_dict['Guardian'])
        nyt = WordCloud(background_color = 'white', width = 800, height = 1800).generate(data_dict['New York Times'])
        wsj = WordCloud(background_color = 'white', width = 800, height = 1800).generate(data_dict['Wall Street Journal'])

        plt.figure(figsize = (9, 6))
        ax1 = plt.subplot2grid((5, 9), (0, 0), rowspan = 5, colspan = 3)
        ax2 = plt.subplot2grid((5, 9), (0, 3), rowspan = 5, colspan = 3)
        ax3 = plt.subplot2grid((5, 9), (0, 6), rowspan = 5, colspan = 3)

        ax1.imshow(guardian)
        ax1.axis('off')
        ax1.set_title('Guardian', y = 1.02, fontsize = 18)

        ax2.imshow(nyt)
        ax2.axis('off')
        ax2.set_title('New York Times', y = 1.02, fontsize = 18)		

        ax3.imshow(wsj)
        ax3.axis('off')
        ax3.set_title('Wall Street Journal', y = 1.02, fontsize = 18)		

        plt.tight_layout() 
        plt.savefig('figures/source_within_topic' + str(topic) + '.png') 
        plt.close() 
        return 
    except:
        pass 

## plot topics by source 
tfidf = TfidfVectorizer(stop_words = augmented_stop_words, max_features = 500)
count = CountVectorizer(stop_words = augmented_stop_words, max_features = 500)

for i in xrange(nmf.k):
	nmf.plot_topic(i)  # topics 
	topic_data = data.query('topic == {0}'.format(str(i)))  # look within topics 
	tfidf_features = tfidf.fit_transform(topic_data['text'])
	tfidf_names = tfidf.get_feature_names()
	tfidf_terms = pd.DataFrame(tfidf_features.toarray(), index = topic_data['source'])
	tfidf_terms = tfidf_terms.groupby(tfidf_terms.index.values).mean() 
	count_features = count.fit_transform(topic_data['text'])
	count_names = count.get_feature_names() 
	count_terms = pd.DataFrame(count_features.toarray(), index = topic_data['source'])
	count_terms = count_terms.groupby(count_terms.index.values).mean() 
	t_convert = scale_conversion(tfidf_terms, (1, 1000))
	c_convert = scale_conversion(count_terms, (1, 1000))
	string_dict = cloud_word_string(t_convert, tfidf_names, c_convert, count_names)
	within_label_cloud_by_topic(string_dict, i)

## document level sentiment 
## template : https://discountllamas.wordpress.com/2011/03/28/basic-sentiment-analysis-of-news-articles/
## WordNet : http://wordnet.princeton.edu/
## http://sentiwordnet.isti.cnr.it/

def sentiment(content):
    wordnet.sentiment.load()
    relevant_types = ['JJ', 'VB', 'RB'] # adjectives, verbs, adverbs
    score = 0
    sentences = split(parse(content, lemmata=True))
    for sentence in sentences:
        for word in sentence.words:
            if word.type in relevant_types:
                pos, neg, obj = wordnet.sentiment[word.lemma]
                score = score + ((pos - neg) * (1 - obj)) # weight subjective words
    return score 

content = data['text'].tolist()
data['sentiment'] = pd.Series([sentiment(x) for x in content])

for i in xrange(nmf.k):
    topic_data = data.query('topic == {0}'.format(str(i)))
    topic_data = topic_data[['source','sentiment']].groupby('source').mean().reset_index()
    sns.barplot(x='source', y='sentiment', data=topic_data)
    plt.xlabel(''); plt.ylabel('') 
    plt.tight_layout() 
    plt.savefig('figures/source_sentiment_topic' + str(i) + '.png') 
    plt.close() 
