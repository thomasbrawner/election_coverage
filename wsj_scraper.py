## ----------------------------------------------------- ##

import itertools 
import json 
import time 
from bs4 import BeautifulSoup 
from pymongo import MongoClient
from selenium import webdriver

## ----------------------------------------------------- ##
## get login info 

with open('api_keys.json') as f:
	login_data = json.load(f)['wsj_login']

USERNAME = login_data[0]['user_name']
PASSWORD = login_data[0]['password']

## ----------------------------------------------------- ##
## log in to wsj

driver = webdriver.Firefox() 
driver.get('https://id.wsj.com/access/pages/wsj/us/signin.html?url=http%3A%2F%2Fwww.wsj.com&mg=id-wsj')

user = driver.find_element_by_name('username')
user.click()
user.send_keys(USERNAME)

pwrd = driver.find_element_by_name('password')
pwrd.click()
pwrd.send_keys(PASSWORD)

driver.find_element_by_id('submitButton').click()

time.sleep(10)

## ----------------------------------------------------- ##
## content scraper
## return dict of source ('WSJ'), web url, timestamp, author, text 

def wsj_content_scraper(article_url):

	out = {'source' : 'WSJ',
		   'web_url' : article_url}

	driver.get(article_url)
	page = BeautifulSoup(driver.page_source)
	
	try:
		ts = page.find('time', attrs = {'class' : 'timestamp'}).get_text(strip = True)
		out['timestamp'] = ts 
	except:
		out['timestamp'] = ''
		pass 

	try:
		body = page.find('div', attrs = {'itemprop' : 'articleBody'})
		out['text'] = body.get_text(strip = True)
	except:
		out['text'] = ''
		pass 

	try:
		author = page.find('div', attrs = {'class' : 'byline'})
		author = author.find('span', attrs = {'itemprop': 'name'}).get_text(strip = True)
		out['author'] = author
	except:
		out['author'] = ''
		pass 

	return out 

## ----------------------------------------------------- ##
## read in urls

with open('wsj_article_urls.txt','rb') as f:
	relevant_urls = json.loads(f.read())

## ----------------------------------------------------- ##
## scrape content and put in mongo 

client = MongoClient()
db = client['election2015']
tab = db['articles']

for url in relevant_urls:
	content = wsj_content_scraper(url)
	if not tab.find_one({'web_url' : content['web_url']}):
		tab.insert_one(content)

client.close() 

## ----------------------------------------------------- ##
## ----------------------------------------------------- ##

