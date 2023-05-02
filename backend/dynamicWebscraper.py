from bs4 import BeautifulSoup
import json
import numpy as np
import requests
from requests.models import MissingSchema
import spacy
import trafilatura
from urllib.parse import urlparse
from string import punctuation
from collections import Counter
import time


def beautifulsoup_extract_text_fallback(response_content):
    
    '''
    This is a fallback function, so that we can always return a value for text content.
    Even for when both Trafilatura and BeautifulSoup are unable to extract the text from a 
    single URL.
    '''
    
    # Create the beautifulsoup object:
    soup = BeautifulSoup(response_content, 'html.parser')
    
    # Finding the text:
    text = soup.find_all(text=True)
    
    # Remove unwanted tag elements:
    cleaned_text = ''
    blacklist = [
            '[document]',
            'noscript',
            'header',
            'html',
            'meta',
            'head', 
            'input',
            'script',
            'style',]

    # Then we will loop over every item in the extract text and make sure that the beautifulsoup4 tag
    # is NOT in the blacklist
    for item in text:
        if item.parent.name not in blacklist:
            cleaned_text += '{} '.format(item)
            
    # Remove any tab separation and strip the text:
    cleaned_text = cleaned_text.replace('\t', '')
    return cleaned_text.strip()

def extract_text_from_single_web_page(url):
    try:
        downloaded_url = trafilatura.fetch_url(url)
        try:
            a = trafilatura.extract(downloaded_url, output_format="json", with_metadata=True, include_comments = False,
                                date_extraction_params={'extensive_search': True, 'original_date': True})
        except AttributeError:
            a = trafilatura.extract(downloaded_url, output_format="json", with_metadata=True,
                                date_extraction_params={'extensive_search': True, 'original_date': True})
        if a:
            json_output = json.loads(a)
            return json_output['text']
        else:
            try:
                resp = requests.get(url, timeout=20)

                # We will only extract the text from successful requests:
                if resp.status_code == 200:
                    return beautifulsoup_extract_text_fallback(resp.content)
                else:
                    # This line will handle for any failures in both the Trafilature and BeautifulSoup4 functions:
                    return np.nan
            # Handling for any URLs that don't have the correct protocol
            except MissingSchema:
                return np.nan
    except Exception as e: 
        print(e)
        try:
            resp = requests.get(url, timeout=20)

            # We will only extract the text from successful requests:
            if resp.status_code == 200:
                return beautifulsoup_extract_text_fallback(resp.content)
            else:
                # This line will handle for any failures in both the Trafilature and BeautifulSoup4 functions:
                return np.nan
        # Handling for any URLs that don't have the correct protocol
        except:
            return np.nan
    
def get_hotwords(text, nlp):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] # 1
    doc = nlp(text.lower()) # 2
    for token in doc:
        # 3
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        # 4
        if(token.pos_ in pos_tag):
            result.append(token.text)
                
    return result # 5

        

def dyanmicScrape(urls):
    text_content = []
    for index, url in enumerate(urls):
        try:
            text_content.append(extract_text_from_single_web_page(url['link']))
        except:
            text_content.append(np.nan)

    cleaned_textual_content = [text for text in text_content if str(text) != 'nan']

    nlp = spacy.load("en_core_web_sm")

    nlp.max_length = 100000000

    data = []

    for index , cleaned_text in enumerate(cleaned_textual_content):

        doc = nlp(cleaned_text)

        domain = urlparse(urls[index]['link']).netloc

        wordCount = len(doc)

        sentenceCount = len(list(doc.sents))

        output = set(get_hotwords(cleaned_text, nlp))

        keywords = [(x[0]) for x in Counter(output).most_common(5)]

        returnData = {"domain": domain, "title": urls[index]['title'], "link": urls[index]['link'], "text": cleaned_text, "wordCount": wordCount, "sentenceCount": sentenceCount, "keywords": keywords}

        data.append(returnData)

    return data

