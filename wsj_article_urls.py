import itertools 
import json 
from robobrowser import RoboBrowser


## search terms
terms = ['chafee', 'clinton', "o'malley", 'sanders', 'webb', 'warren',
         'bush', 'carson', 'christie', 'cruz', 'fiorina', 'gilmore', 'graham',
         'huckabee', 'jindal', 'kasich', 'pataki', 'paul', 'perry', 'rubio', 'santorum',		 
         'trump', 'walker', 'romney', 'election', 'presidential', 'cycle', 'primary',
         'primaries', 'candidate', 'race']
		 
## dates to search in 2015
months, days = range(1, 9), range(1, 32)
dates = itertools.product(months, days)

## search the archives for potentially relevant material
browser = RoboBrowser(history = True)
relevant_urls = []
bad_urls = []
for date in dates:
    m, d = date[0], date[1]
    archive_url = 'http://www.wsj.com/public/page/archive-2015-' + str(m) + '-' + str(d) + '.html'
    try:
        browser.open(archive_url)
        articles = browser.find_all('h2')
        for article in articles:
            if any(word in article.get_text().lower() for word in terms):
                relevant_urls.append(article.find('a').get('href'))
    except:
        bad_urls.append(archive_url)
        pass 

## save the urls 
with open('wsj_article_urls.txt','rb') as f:
    f.write(json.dumps(relevant_urls))
    f.close() 
