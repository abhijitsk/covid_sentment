from django.apps import AppConfig
from django.conf import settings
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model








class HomeConfig(AppConfig):

    name = 'home'
    path = os.path.join(settings.MODELS,'models.p')
    path2 = os.path.join('F:\\website\\tweets_download\\twitter\\home\\models\\','models_RF.p')
    path3 = os.path.join('F:\\website\\tweets_download\\twitter\\home\\models\\','model_full_features.h5')
    path4 = os.path.join('F:\\website\\tweets_download\\twitter\\home\\models\\','tokenizer_lstm')

    with open(path,'rb') as pickled:
        classifiers = pickle.load(pickled)

    with open(path2,'rb') as pickled:
        Random_Forrest = pickle.load(pickled)

    with open(path4,'rb') as pickled:
        LSTM_tokenizer = pickle.load(pickled)
    Random_Forrest = Random_Forrest['model_Random']
    Naive_bayes = classifiers['Naive']
    SVM = classifiers['SVM']
    count_vect = classifiers['count_vect']
    tfidf_vect = classifiers['tf_idf_vect']
    LSTM = load_model(path3)



    
