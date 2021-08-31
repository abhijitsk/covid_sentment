
import tweepy
from pymongo import MongoClient
import json
import sys
from .twitter_credentials import *
#from twitter_credentials import *
import pandas as pd
from .streamer1 import *


tweets_for_website = []

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth)
for tweet in tweepy.Cursor(api.search, q = ("#corona"),count = 100, lang= 'en').items(100):
        tweets_for_website.append(tweet.text)

tweets_pandas = pd.DataFrame(tweets_for_website, columns = ['liveTweets'])
tweets_pandas['preprocessed'] = tweets_pandas['liveTweets'].apply(preprocess)



