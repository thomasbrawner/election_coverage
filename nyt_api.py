import json 
from nytimesarticle import articleAPI


with open('api_keys.json') as f:
	api_key = json.load(f)['nyt_api_key']
api = articleAPI(api_key)

## ------------------------------------------------------------------ ##
## set up date ranges to search 

date_list = [('20150101', '20150131'),
             ('20150201', '20150228'),
             ('20150301', '20150331'),
             ('20150401', '20150430'),
             ('20150501', '20150531'),
             ('20150601', '20150630'),
             ('20150701', '20150731'),
             ('20150801', '20150831')]

## ------------------------------------------------------------------ ##
## get article data

articles_out = []
    for date_tuple in date_list:
        for i in range(0, 100):
            articles = api.search(q='election', 
                                  fq={'source' : ['The New York Times']}, 
                                  begin_date=date_tuple[0], 
                                  end_date=date_tuple[1],
                                  page=str(i))
            articles_out += articles['response']['docs']

with open('nyt_results_election_2015.txt', 'wb') as f:
    json.dump(articles_out, f)
    f.close() 
