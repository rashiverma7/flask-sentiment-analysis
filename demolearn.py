import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
nltk.download('stopwords')

def loadData(product_id):
    

    df = pd.read_csv('./static/test_data.csv')
    df.head()
    df = df.iloc[0:50000, :]
    df.shape

    df = df[['Reviews', 'Rating']]
    df = df.dropna()    
    df.head()

    df = df[df['Rating'] != 3]
    df = df.reset_index(drop=True)
    df.info()

    df['sentiment'] = np.where(df['Rating'] > 3, 1, 0)
    df.head()

    
    X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['sentiment'], test_size=0.2, random_state=0)

    

    def cleanText(raw_text, remove_stopwords=True, stemming=False, split_text=False):

        letters_only = re.sub("[^a-zA-Z]", " ", raw_text)
        words = letters_only.lower().split()

        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]

        if stemming == True:
            stemmer = SnowballStemmer('english')
            words = [stemmer.stem(w) for w in words]

        if split_text == True:
            return (words)

        return (" ".join(words))

    X_train_cleaned = []
    X_test_cleaned = []

    for d in X_train:
        X_train_cleaned.append(cleanText(d))

    for d in X_test:
        X_test_cleaned.append(cleanText(d))

    X_train_cleaned[10]

    
    countVect = CountVectorizer()
    X_train_countVect = countVect.fit_transform(X_train_cleaned)
    print("Number of features : %d \n" % len(countVect.get_feature_names()))  # 6378
    print("Show some feature names : \n", countVect.get_feature_names()[::1000])

    
    mnb = MultinomialNB()
    mnb.fit(X_train_countVect, y_train)

    

    def modelEvaluation(predictions):
        print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test, predictions)))
        print("\nAUC score : {:.4f}".format(roc_auc_score(y_test, predictions)))
        print("\nClassification report : \n", metrics.classification_report(y_test, predictions))
        print("\nConfusion Matrix : \n", metrics.confusion_matrix(y_test, predictions))

    predictions = mnb.predict(countVect.transform(X_test_cleaned))
    modelEvaluation(predictions)

    pos_count = 0
    neg_count = 0
    for i in range(len(predictions)):
        if (predictions[i] == 1):
            pos_count = pos_count + 1
        elif (predictions[i] == 0):
            neg_count = neg_count + 1

    print('Positive reviews: ', pos_count)
    print('Negative reviews: ', neg_count)

    total = pos_count + neg_count
    pos_per = pos_count / float(total)
    neg_per = neg_count / float(total)

    print('Total: ', total)
    print('Positive percent: ',pos_per)
    print('Negative percent: ',neg_per)
    
    return pos_per*100, neg_per*100



    