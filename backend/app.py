from flask import Flask, request, jsonify
from flask_cors import CORS
from sentimentAnalysis3 import *
from webscraper import webscraper
from emotionAnalysis2 import *
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return '<h1>Test</h1>'


@app.route('/api/analyses', methods=['POST', 'GET'])
def analyses():
    try:
        req = request.json

        search = req['request']
        df = webscraper(search)

        if df.empty:
            df = pd.read_json("./Mariodata.json")

        # df = pd.read_json("./Mariodata.json")
        df["sentiment"] = ""
        df["emotion"] = ""

        for index, row in df.iterrows():
            sentiment = SentimentAnalysis(row['text'])
            df.at[index, 'sentiment'] = sentiment

            emotion = emotionAnalysis(row['text'])
            df.at[index, 'emotion'] = emotion

        response = {
            "status": "success",
            "message": search,
            "data": df.to_dict(orient='records')
            }

        return jsonify(response)
    except Exception as e:
        print(e)
        response = {
            "status": "fail",
            }
        return jsonify(response)
    
@app.route('/api/keywordAnalyses/<string:Hash>', methods=['POST', 'GET'])
def keyword(Hash):
    try:
        print(Hash)
        search = Hash

        tweets_df = webscraper(search)

        tweets_df["sentiment"] = ""
        tweets_df["emotion"] = ""

        for index, row in tweets_df.iterrows():
            sentiment = SentimentAnalysis(row['tweet'])
            tweets_df.at[index, 'sentiment'] = sentiment

            emotion = emotionAnalysis(row['tweet'])
            tweets_df.at[index, 'emotion'] = emotion

        response = {
            "status": "success",
            "message": search,
            "data": tweets_df.to_dict(orient='records')
            }

        return jsonify(response)
    except Exception as e:
        print(e)
        response = {
            "status": "fail",
            }
        return jsonify(response)

@app.route('/api/message', methods=['POST', 'GET'])
def message():
    return '<h1>To be implemented, should return a response to the message prompt from the user</h1>'

if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()