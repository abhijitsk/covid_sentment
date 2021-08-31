from django.apps import AppConfig
from django.conf import settings
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model








class HomeConfig(AppConfig):

    name = 'home'
    print(settings.MODELS)
    path = os.path.join(settings.MODELS,'models.p')
    


    with open(path,'rb') as pickled:
        classifiers = pickle.load(pickled)

    Naive_bayes = classifiers['Naive']
    SVM = classifiers['SVM']
    count_vect = classifiers['count_vect']
    tfidf_vect = classifiers['tf_idf_vect']

    
