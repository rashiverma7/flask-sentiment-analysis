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

#nltk.download('stopwords')


def loadData(product_id):
    df = pd.read_csv("./static/Amazon_Unlocked_Mobile.csv")
    dftest = pd.read_csv("./static/test_data.csv")
    df.head()
    df = df.iloc[:,:]
    df=df[['Reviews','Rating']]
    df=df.dropna()
    df.head()

    df=df[df['Rating']!=3]
    df=df.reset_index(drop=True)
    df.info()

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
    

    df['sentiment']=np.where(df['Rating'] > 3, 1, 0)
    df.head()
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

    for d in X_train:
        X_train_cleaned.append(cleanText(d))
    
    for d in X_test:
        X_test_cleaned.append(cleanText(d))



    CVect = CountVectorizer() 
    X_train_countVect = CVect.fit_transform(X_train_cleaned)


    mnb = MultinomialNB()
    mnb.fit(X_train_countVect, y_train)



    def modelEvaluation(predictions):
        print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test, predictions)))
        print("\nAUC score : {:.4f}".format(roc_auc_score(y_test, predictions)))
        print("\nClassification report : \n", metrics.classification_report(y_test, predictions))
        print("\nConfusion Matrix : \n", metrics.confusion_matrix(y_test, predictions))

    predictions = mnb.predict(CVect.transform(X_test_cleaned))
    modelEvaluation(predictions)


    tfidf = TfidfVectorizer(min_df=5)
    X_train_tfidf = tfidf.fit_transform(X_train)


    lr = LogisticRegression()
    lr.fit(X_train_tfidf, y_train)

    print("Number of features : %d \n" %len(tfidf.get_feature_names())) 
    print("Show some feature names : \n", tfidf.get_feature_names()[::1000])

    predictions = lr.predict(tfidf.transform(X_test_cleaned))
    modelEvaluation(predictions)
