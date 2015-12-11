import json 
import numpy as np 
import requests
import time 
from bs4 import BeautifulSoup 
from pymongo import MongoClient


with open('nyt_results_election_2015.txt') as f:
    data = json.loads(f.read())

bad_url = []
for article in data:
    url = article['web_url']
    try:
        r = requests.get(url)
        s = BeautifulSoup(r.content)
        c = s.find_all('p', attrs = {'itemprop' : 'articleBody'})
        article['text'] = '\n\n'.join([x.get_text() for x in c])
    except:
        print 'error'
        bad_url.append(url)
    if np.random.choice(np.arange(2), p = [0.95, 0.05]) == 1:
        print article
    time.sleep(2)

client = MongoClient(port = 27018)
db = client['election2015']
tab = db['articles']

for article in data:
    if not tab.find_one({'_id' : article['_id']}):
        tab.insert_one(article)
