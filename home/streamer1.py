import pandas as pd 
import string
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np 
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
import nltk



def process_forKeras(predictions):
    check_label = ['anger','fear','joy','sad']
    predictions_list = [check_label[np.argmax(text)] for text in predictions]
    


    return predictions_list





def get_graph():
    buffer  = BytesIO()
    plt.savefig(buffer, format = 'png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def plotting(predictions, title_name):
    data_for_cloud = np.unique(predictions, return_counts =True)
    x = data_for_cloud[1].astype(np.int)
    y = data_for_cloud[0]
    
    
    
    plt.switch_backend('AGG')
    plt.pie(x, autopct = '%1.0f%%')
    plt.legend(y)
    plt.title(title_name)
    graph = get_graph()
    return graph

def plot_wordcloud(complete_words):

    wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119, max_words = 30  ).generate(complete_words)
    plt.switch_backend('AGG')
    plt.imshow(wordCloud, interpolation = "bilinear") # need to explain this line of code
    plt.axis('off')
    plt.title('WORD CLOUD', size = 20, color = "blue")
    graph = get_graph()
        
    return graph



def preprocess(tweet):
    STOPWORDS = set(stopwords.words('english') + list(string.punctuation) + ['covid','pandemic', 'virus', 'corona','rt'])
    tweet = tweet.lower() # convert to lower case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ' ,tweet) # replace the urls in the link with space
    tweet = re.sub('@[^\s]+',' ',tweet) # replace the @ symabol with space
    tweet = re.sub(r'#([^\s]+)',r'\1',tweet) # replace the # with the word
    tweet = re.sub('[^A-Za-z\s]',' ', tweet) # try and check if its only alphates
    tweet = word_tokenize(tweet) # using NLTK to tokenize
    st = nltk.PorterStemmer() # calling an instance of  porter stemmer
    tweet = [st.stem(word) for word in tweet]
    lm = nltk.WordNetLemmatizer()  #calling an instance of lemmatizer 
    tweet = [lm.lemmatize(word) for word in tweet]
    
    
    return  ' '.join([word for word in tweet if word not in STOPWORDS])





