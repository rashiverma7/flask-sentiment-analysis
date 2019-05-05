import re
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

#nltk.download('stopwords')




def loadData(product_id):

    df = pd.read_csv('./static/train_data.csv')
    df.head()
    df = df.iloc[0:5000,:]
    df.shape

    df = df[['Reviews', 'Rating']]
    df.dropna()
    df.head()

    df = df[df['Rating']!=3]
    df = df.reset_index(drop=True)
    df.info()

    df['sentiment']=np.where(df['Rating'] > 3, 1, 0)
    df.head()

    dftest = pd.read_csv('./static/test_data.csv')
    dftest.head()
    dftest = dftest.iloc[0:50000,:]
    dftest.shape

    dftest = dftest[['Reviews','Rating']]
    dftest.dropna()
    dftest.head()

    dftest = dftest[dftest['Rating']!=3]
    dftest = dftest.reset_index(drop=True)
    dftest.info()

    if(product_id == 1):
        dftest = dftest.iloc[0:10000,:]
    

    elif(product_id == 2):
        dftest = dftest.iloc[10001:20000, :]
    

    elif(product_id == 3):
        dftest = dftest.iloc[20001:30000, :]
    

    elif(product_id == 4):
        dftest = dftest.iloc[30001:40000, :]

    elif(product_id == 5):
        dftest = dftest.iloc[40001:50000, :]
    

    
    dftest['sentiment'] = np.where(dftest['Rating'] > 3, 1, 0)
    dftest.head()

    #X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['sentiment'], test_size=0.2, random_state=0)

    X_train = df['Reviews']
    y_train = df['sentiment']

    X_test = dftest['Reviews']
    y_test = dftest['sentiment']

    def cleanText(raw_text, remove_stopwords=True, stemming=True, split_text=False):
    
        letters_only = re.sub("[^a-zA-Z]", " ", raw_text)  
        words = letters_only.lower().split()  
    
        if remove_stopwords: 
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        
        if stemming==True: 
            stemmer = SnowballStemmer('english') 
            words = [stemmer.stem(w) for w in words]
        
        if split_text==True: 
            return (words)
    
        return( " ".join(words))

    X_train_cleaned = []
    X_test_cleaned = []


    train_start = time.time()
    try:
        for d in X_train:
            if type(d) is str:
                X_train_cleaned.append(cleanText(d))
            else:
                X_train_cleaned.append(cleanText(''))
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print 'In train at ' + d + message
    train_end = time.time()
    print 'Time taken to train: ', (train_end - train_start)

    test_start = time.time()
    try:
        for d in X_test:
            if type(d) is str:
                X_test_cleaned.append(cleanText(d))
            else:
                X_test_cleaned.append(cleanText(''))
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print 'In test at ' + d + message
    test_end = time.time()
    print 'Time taken to test: ', str(test_end - test_start)



    CVect = CountVectorizer(stop_words=None) 
    X_train_countVect = CVect.fit_transform(X_train_cleaned)


    mnb = MultinomialNB()
    mnb.fit(X_train_countVect, y_train)



    def modelEvaluation(predictions):
        print ("Accuracy on validation set: {:.4f}".format(accuracy_score(y_test, predictions)))
        #print("Classification Report: {:.4f}".format(metrics.classification_report(y_test, predictions)) )
        #print("\nAUC score : {:.4f}".format(roc_auc_score(y_test, predictions)))
        print("Classification report : ") 
        print(metrics.classification_report(y_test, predictions))
        print("Confusion Matrix: ")
        print(metrics.confusion_matrix(y_test, predictions))

    predictions = mnb.predict(CVect.transform(X_test_cleaned))
    modelEvaluation(predictions)

    pos_count = 0
    neg_count = 0
    for i in range(len(predictions)):
        if(predictions[i] == 1):
            pos_count = pos_count + 1
        elif(predictions[i] == 0):
            neg_count = neg_count + 1

    print('Positive reviews: ', pos_count)
    print('Negative reviews: ', neg_count)

    pos_per = pos_count/float(pos_count+neg_count)
    neg_per = neg_count/float(pos_count+neg_count)

    print('Positive percent: ' + str(pos_per*100) + "%")
    print('Negative percent: ' + str(neg_per*100) + "%")

    return pos_per*100, neg_per*100
    



    '''tfidf = TfidfVectorizer(min_df=5)
    X_train_tfidf = tfidf.fit_transform(X_train)


    lr = LogisticRegression()
    lr.fit(X_train_tfidf, y_train)

    print("Number of features : %d \n" %len(tfidf.get_feature_names())) 
    print("Show some feature names : \n", tfidf.get_feature_names()[::1000])

    predictions = lr.predict(tfidf.transform(X_test_cleaned))
    modelEvaluation(predictions)

    return pos_per*100, neg_per*100'''
    