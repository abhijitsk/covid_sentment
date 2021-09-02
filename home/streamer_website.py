
import tweepy
from pymongo import MongoClient
import json
import sys
import pandas as pd
from .streamer1 import *
import os

tweets_for_website = []

auth = tweepy.OAuthHandler(os.environ.get('consumer_key'), os.environ.get('consumer_secret'))
auth.set_access_token(os.environ.get('access_token_key'), os.environ.get('access_token_secret'))

api = tweepy.API(auth)
for tweet in tweepy.Cursor(api.search, q = ("#corona OR #pandemic OR #COVID"+"-filter:retweets"),count = 200, lang= 'en',tweet_mode = 'extended').items(100):
        tweets_for_website.append(tweet.full_text)

tweets_pandas = pd.DataFrame(tweets_for_website, columns = ['liveTweets'])
tweets_pandas['preprocessed'] = tweets_pandas['liveTweets'].apply(preprocess)



