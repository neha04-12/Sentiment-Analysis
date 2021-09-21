# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 17:03:42 2021

@author: SREENEHA
"""
import pandas as pd
#import numpy as np
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import GaussianNB
from sklearn import svm



def read_csvfile ():
#open and read into list
#return list
    dftrain = pd.read_csv("traindata.csv")
    dftrain.columns = ["id","label","review"]
    trainreview = dftrain["review"].tolist()
    trainid = dftrain["id"].tolist()
    trainlabel = dftrain["label"].tolist()
    dftest = pd.read_csv("testdata.csv")
    dftest.columns = ["id","review"]
    testid = dftest["id"].tolist()
    testreview = dftest["review"].tolist()
    
    return trainreview, trainid, trainlabel, testid, testreview, dftrain["review"], dftest["review"]
    
    

def NLTK_proccessed(listrec):
#return string
    redundantchar=['@','#','!',"%",":",";","/","'",".",",","*","$","...","-","(",")"]
#stop words
    stop_words =set(stopwords.words("english"))
    #print(stop_words)
    stopsentences = []
    for sentence in listrec:
        stopfiltered_sentence= []
        tokenized_sentence=word_tokenize(sentence)
        for word in tokenized_sentence:
            if word not in stop_words and word not in redundantchar:
                stopfiltered_sentence.append(word)
        stopsentences.append(stopfiltered_sentence)
    
    
#stemming

    ps = PorterStemmer()
    stemmedsentences=[]
    for sentence in stopsentences:
        stemfiltered_sentence=[]
        for word in sentence:
            stemfiltered_sentence.append(ps.stem(word))
        stemmedsentences.append(stemfiltered_sentence)
    
#Lemmetization
    
    lem = WordNetLemmatizer()
    lemsentences=[]
    for sentence in stemmedsentences:
        lemfiltered_sentence=[]
        for word in sentence:
            lemfiltered_sentence.append(lem.lemmatize(word,"v"))
        lemsentences.append(lemfiltered_sentence)
    
#parts of speech (POS tagging)
    redundantpos=['CD','DT','NNS','NNP','NNPS','EX']
    possentences=[]
    for sentence in lemsentences:
        filteredpos_sentence=[]
        possentence = nltk.pos_tag(sentence)
        #print("pos", possentence)
        for record in possentence:
            if record[1] not in redundantpos:
                filteredpos_sentence.append(record[0])    
        possentences.append(filteredpos_sentence)    
    return (possentences)

def create_dict (finalfilteredwords):
#return dictionary
    dictionaryofn =[]
    for sentence in finalfilteredwords:
        for word in sentence:
            dictionaryofn.append(word)
    freqdist = FreqDist(dictionaryofn)
    #print("freqdist", freqdist)
    finaldictionary=freqdist.most_common(250)
    #IDF calculation
    
    totaldocs=len(finalfilteredwords)
    IDFarray=[]
    
    for word in finaldictionary:
        count=0
        for sentence in finalfilteredwords:
            if word[0] in sentence:
                count=count+1
            #else:
                #print(word, sentence)
        idf=math.log(totaldocs/count)
        IDFarray.append(idf)
    
    return finaldictionary, IDFarray

def create_vector(trainreview, dictionary, IDFarray):
#return frequency table of dictionary size
# existance
    traindictionary=[]
    i=0
    for sentence in trainreview:
        wordvectrain=[None]*250
        TFIDF=[None]*250
        for i in range(len(dictionary)):
            word = dictionary[i][0]
            #if word in sentence:
            wordvectrain[i] = sentence.count(word)
            TFIDF[i]=wordvectrain[i]*IDFarray[i]
            #else:
                #wordvectrain[i] = 0 
        traindictionary.append(TFIDF)        
    return traindictionary


# no. of times occurance
#TF/IDF
    #tf=TfidfVectorizer()
    #text_tf= tf.fit_transform(dftrain)
    #return text_tf[1]
    
    


#calling read_csvfile
trainreview, trainid, trainlabel, testid, testreview, dataframeoftrainreview, dataframeoftestreview = read_csvfile()
#print(testid[:10])
#print(testreview[:10])
 
#zipping of columns   
#testallrec = tuple(zip(testid, testreview))


#calling NLTK_processed for train
filteredtrainreview = NLTK_proccessed(trainreview)
#print(trainreview[1])
#print(filteredtrainreview[1])

#calling dictionary creation
dictionary, IDFarray = create_dict(filteredtrainreview)
#print(IDFarray)

#calling frequency table of train data
finaltraindata=create_vector(filteredtrainreview, dictionary, IDFarray)
#print(finaltraindata[:5])

#calling NLTK_processed for test
filteredtestreview = NLTK_proccessed(testreview)

#calling frequency table of test data
finaltestdata=create_vector(filteredtestreview, dictionary, IDFarray)

#training model
#model = GaussianNB()
model = svm.SVC()
model.fit(finaltraindata, trainlabel)

#predict model
predicted= model.predict(finaltestdata) # 0:Overcast, 2:Mild
#print (predicted)
#print (trainlabel[8])

dfpredicted=pd.DataFrame(predicted)

dfpredicted.to_csv('testSVCprediction.csv')
#matched=0
#unmatched=0
#for i in range (len(predicted)):
    #if predicted[i]==trainlabel[i]:
        #matched=matched+1
    #else:
        #unmatched=unmatched+1
#print(matched)
#print(unmatched) 

#accuracy= (matched/(matched+unmatched))*100
#print(accuracy)