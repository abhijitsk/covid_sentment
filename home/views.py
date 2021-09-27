from django.shortcuts import render
#from . import dbMongo
#from pymongo import MongoClient
from . import streamer_website
import json
import sys
import pandas as pd
from .apps import HomeConfig
from rest_framework.views import APIView
from django.http import JsonResponse, HttpResponse
from keras.preprocessing.sequence import pad_sequences
from . import streamer1
from . import streamer_website
import tweepy




# Create your views here.

def home(request):
    file_home = streamer_website.tweets_pandas
    home_tweets = file_home['liveTweets']
    pre_home = file_home['preprocessed']

    
    return render(request,'home.html',{'tweet':home_tweets,'preprocessed':pre_home})




class call_model(APIView):
    
    def get(self, request):
        if request.method == 'GET':
            tweets_for_website =[]

            api = streamer_website.api
            
            for tweet in tweepy.Cursor(api.search, q = ("#corona OR #pandemic OR #COVID"+"-filter:retweets"),count = 200, 
                                                                        lang= 'en', tweet_mode = 'extended').items(100):
                tweets_for_website.append(tweet.full_text)
                

            tweets_pandas = pd.DataFrame(tweets_for_website, columns=['liveTweets'])
            tweets_pandas.drop_duplicates()
            tweets_pandas['preprocessed'] = tweets_pandas['liveTweets'].apply(streamer1.preprocess)

            text =tweets_pandas['preprocessed'].tolist()
            text_original = tweets_pandas['liveTweets'].tolist()
            

            
            vector = HomeConfig.count_vect.transform(text)
            test_tfidf = HomeConfig.tfidf_vect.transform(vector)
            test_LSTM = HomeConfig.LSTM_tokenizer.texts_to_sequences(text)
            test_LSTM = streamer1.pad_sequences(test_LSTM,maxlen= 31, padding='post')
           
            prediction = HomeConfig.Naive_bayes.predict(test_tfidf)
            prediction_svm = HomeConfig.SVM.predict(test_tfidf)
            prediction_RF = HomeConfig.Random_Forrest.predict(test_tfidf)
            prediction_LSTM = HomeConfig.LSTM.predict(test_LSTM)
            prediction_LSTM1 = streamer1.process_forKeras(prediction_LSTM)
           

            
    
            
            outfile = pd.DataFrame(prediction, columns = ['Naive_bayes'])
            outfile['SVM'] = prediction_svm
            outfile['Rand_forrest'] = prediction_RF
            outfile['ANN_LSTM'] = prediction_LSTM1
            
            outfile['text'] = text
            outfile['text_original'] = text_original
            
            json_file = outfile.to_json()

            data = json.loads(json_file)
            data2 = outfile.to_dict(orient = 'records' )
            

    

            pie_chart_naive = streamer1.plotting(prediction.tolist(),'Naive Bayes')
            pie_chart_RF = streamer1.plotting(prediction_RF.tolist(), 'Random Forrest')
            

    
            pie_chart_SVM = streamer1.plotting(prediction_svm.tolist(), 'SVM')
            pie_chart_LSTM = streamer1.plotting(prediction_LSTM1,'LSTM')
            #wordplot2 = streamer1.plot_wordcloud(' '.join(text)) # features names

            #pie_chart_LSTM = streamer1.plotting(predictionLSTM, )
            #wordplot3 = streamer1.plot_wordcloud(' '.join(complete_words_LSTM))

            
            
            response = {'data':data, 
                        'data2':data2,
                        'pie_chart_naive':pie_chart_naive,
                        'pie_chart_RF':pie_chart_RF,
                        'pie_chart_SVM':pie_chart_SVM,
                        'pie_chart_LSTM':pie_chart_LSTM}

        



            
            return render(request,'predictions.html',response)
