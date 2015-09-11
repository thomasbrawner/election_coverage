## ----------------------------------------------------- ##

import json 
import requests
import time 
from bs4 import BeautifulSoup 
from pymongo import MongoClient

## ----------------------------------------------------- ##
## read article data 

with open('guardian_results_us_news_2015.txt') as f:
	data = json.loads(f.read())

## ----------------------------------------------------- ##
## scrape and store article content

bad_url = []
for article in data:
	url = article['webUrl']
	try:
		r = requests.get(url)
		s = BeautifulSoup(r.content)
		c = s.find('div', attrs = {"class" : "content__article-body from-content-api js-article__body"})
		article['text'] = c.text.strip()
	except:
		print 'error'
		bad_url.append(url)
	time.sleep(2)

## ----------------------------------------------------- ##
## add source key for the guardian 

for article in data:
	article['source'] = 'Guardian'

## ----------------------------------------------------- ##
## mongo 

client = MongoClient()
db = client['election2015']
tab = db['articles']

for article in data:
	if not tab.find_one({'webUrl' : article['webUrl']}):
		tab.insert_one(article)

## ----------------------------------------------------- ##

""" mongo shell...

> use election2015
switched to db election2015
> db.articles.count()
3271
> db.articles.findOne()
{
	"_id" : ObjectId("55e9e88b8859ae1891022ba3"),
	"sectionName" : "US news",
	"webTitle" : "Duck, duck, goose: judge strikes down California foie gras ban",
	"text" : "Foie gras, the divisive delicacy produced from fatty duck or goose liver, may once again grace the menus of haute restaurants in California.\nUS district court judge Stephen Wilson ruled on Wednesday that a state law banning restaurants from serving foie gras infringed on a federal law which supersedes it. \nThe California law explicitly outlaws force-feeding birds “for the purpose of enlarging the bird’s liver beyond normal size”, the process by which foie gras has been made for thousands of years. It also banned the sale of products from force-fed birds. \nFoie gras is made by force-feeding corn to young ducks and geese. The birds’ metabolism can’t cope with the amount of food, and their livers begin to swell to as much as eight times the normal size. Many animal rights activists call the process cruel and inhumane; producers say the birds are not harmed by it. \nThe California legislature passed the ban in 2004 and then-governor Arnold Schwarzenegger signed it into law that year, but it didn’t take effect until 2012.\nFoie gras producers in Canada and New York, along with local restaurant owners, challenged the ban in court, arguing that it was “unconstitutionally vague” and in violation of the US constitution’s commerce clause.\nIn August 2013 a federal appeals court upheld the law, and in October 2014 the US supreme court declined to review the decision. \nSome restaurants, like San Francisco’s Dirty Habit, are planning to celebrate the ruling by serving foie gras as early as Wednesday night. Dirty Habit announced a special foie gras menu on Twitter. \n\n\n— Dirty Habit (@dirtyhabitsf)\nJanuary 7, 2015\nWant Foie? WE'VE GOT IT! Join us tonight for a 4-course foie gras menu $60... http://t.co/aolYk9srdx #dirtyhabitsf pic.twitter.com/vUaYVlFvow",
	"webUrl" : "http://www.theguardian.com/us-news/2015/jan/07/ban-foie-gras-california-struck-down",
	"id" : "us-news/2015/jan/07/ban-foie-gras-california-struck-down",
	"source" : "Guardian",
	"webPublicationDate" : "2015-01-07T23:57:15Z",
	"type" : "article",
	"apiUrl" : "http://content.guardianapis.com/us-news/2015/jan/07/ban-foie-gras-california-struck-down",
	"sectionId" : "us-news"
}
"""

## ----------------------------------------------------- ##

