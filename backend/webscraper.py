import requests
import urllib
import pandas as pd
from requests_html import HTML
from requests_html import HTMLSession
from dynamicWebscraper import dyanmicScrape
import json
import time

def get_source(url):
    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)
        return

def get_results(query):
    
    query = urllib.parse.quote_plus(query)
    responses = []
    for page in range(1,11):
        responses.append((get_source("https://www.google.co.uk/search?q=" + query + (f"&start={10*(page-1)}" if page > 1 else ""))))
        time.sleep(5)
    
    return responses

def parse_results(responses):
    output = []

    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    css_identifier_link = ".yuRUbf a"

    for response in responses:
    
        results = response.html.find(css_identifier_result)
    
        for result in results:

            item = {
                'title': result.find(css_identifier_title, first=True).text,
                'link': result.find(css_identifier_link, first=True).attrs['href'],
            }
        
            output.append(item)
        
    return output

def google_search(query):
    responses = get_results(query)
    print(responses[0])
    return parse_results(responses)

def webscraper(input):
    print(input)

    results = google_search(input)

    scrapeResults = dyanmicScrape(results)

    resultsDF = pd.DataFrame.from_dict(scrapeResults)

    return resultsDF 

if __name__ == "__main__":

    print('Enter the text you want webscraped')

    userInput = input()

    results = google_search(userInput)
    
    scrapeResults = dyanmicScrape(results)

    with open('data.json', 'w') as f:
        json.dump(scrapeResults, f, indent=4)
