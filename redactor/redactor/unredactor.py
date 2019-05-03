#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
import io
import os
import pdb
import sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from sklearn.datasets import load_files 
from nltk import ngrams

glob_text= "Train data/*.txt"
glob_text2= "Test data/*.txt"

def get_entity(text):
    """Prints the entity inside of the text."""
    
    words=word_tokenize(text)
    wordsC=len(words)
    feature=[]
    featuresfinal=[]
    itemlist=[]
    for chunk in ne_chunk(pos_tag(word_tokenize(text))):
        x=[]
        if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #x=chunk.label(), ' '.join(c[0] for c in chunk.leaves())
            for c in chunk.leaves():
                x.append(c[0])
                #print(x)
            itemlist.append(x)
        
               
               
        name=''
        count=0
        wordlengths=[]
        namewordcount=''
       
    
    for item in itemlist:
        wordlengths=[]
        features=[]
        namewordcount=len(' '.join(item))
        for word in item:
            wordlengths.append(len(word))
        if(len(wordlengths)<3):
            wordlengths.append(0)
            wordlengths.append(0)
        name=' '.join(item) 
        #features.append(wordlengths[0])
        #features.append(wordlengths[1])
        #features.append(wordlengths[2])
        features.append(len(name))
        feature.append(name)
        features.append(wordsC)
        features.append(namewordcount)
        #features=str(features)
        featuresfinal.append(features)           
                     #   y=[]
                    #    y.append(x)
    #print(featuresfinal)
    return feature,featuresfinal

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge 
from sklearn.metrics import precision_recall_fscore_support
def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    xtrain=[]
    ytrain=[]
    x_train=[]
    y_train=[]
    for thefile in glob.glob(glob_text):
        #print(thefile)
        with io.open(thefile, 'r', encoding='utf-8',errors='ignore') as fyl:
            text = fyl.read()
            y_train,x_train=get_entity(text)
            #print(y_train)
            if (len(x_train)==0):
                x_train=[[0,0,0]]
                #y_train=[' ']
            
          
        for train in x_train:
            #print(train)
        #if(x_train[train==[]]):
         #   train=[0,0,0,0,0]
            xtrain.append(train)
    
    #xtrain.append(x_train)
        for ytrain in y_train:
            #print(ytrain)
            #ytrain=''.join(train)
    
    #print(xtrain,ytrain)
    #print(type(y_train))
            #print(xtrain,ytrain)
            return xtrain,y_train
            
            
def training(x,y):
    
        X_train=np.atleast_2d(x)
        #a,b,c=X_train.shape
       # Xtrain=X_train.reshape(a,b*c)
        #print(X_train.shape)
        Y_train=np.array(y)
        #Y_train=Y_train.reshape(1,-1)
        #print(Y_train.shape)

        model=MultinomialNB()
        M=model.fit(X_train,Y_train)
        return M

def predict(Mod,xtest):
    
    x_test=np.atleast_2d(xtest)
    #print(x_test)
    #X_test=x_test.reshape(1,-1)
    #print(x_test.shape)
    result=Mod.predict(x_test)
    #print(result)
    predicted=" "
    predicted= ''.join(result)
    #print(predicted)
    return result,predicted
    
    
    
def precision(d):
    X=[]
    x=[]
    n=d
    #X.append(c)
    #print(X)
    for thefile in glob.glob(glob_text):
        #print(thefile)
        #k=[1,0,1]
        #X=['Future','Shakespear']
        with io.open(thefile, 'r', encoding='utf-8',errors='ignore') as fyl:
            text = fyl.read()
            
        for chunk in ne_chunk(pos_tag(word_tokenize(text))):
        
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #x=chunk.label(), ' '.join(c[0] for c in chunk.leaves())
                for c in chunk.leaves():
                    #y=c[0][0]
                    x.append(c[0])
            
    X=x[:103]
    #print(n)
    prec=precision_recall_fscore_support(X,n,average = 'micro')
    #print(prec)
    return prec



                
def redact(text):
    text2=''
    text3=''
    
    #for thefile in glob.glob(glob_text2):
        #print(thefile)
     #   with io.open(thefile, 'r', encoding='utf-8',errors='ignore') as fyl:
      #      text = fyl.read()
            
    for chunk in ne_chunk(pos_tag(word_tokenize(text))):
        x=[]
        if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #x=chunk.label(), ' '.join(c[0] for c in chunk.leaves())
            for c in chunk.leaves():
                x.append(c[0])
                #print(x)
                #itemlist.append(x)
            #print(x)
        for k in x:
                #print(k)
            text2=text.replace(str(k),"█"*len(str(k)))
            text3=text2
            #print(text3)
            #return x
    #print(text3)
    return text3,x
        
        
def redact_features(red):
    features=[]
    finalfeatures=[]
    Rnamewordcount=''
    RwordsC=''
    Rwordlen=''
    redacted=[]
    Rwordlengths=[]
    words=word_tokenize(str(red))
    
    RwordsC=len(words)
    for i in words:
        if (i=="█"*len(i)):
            Rwordlen=len(i)
            redacted.append(i)
        for n in redacted:
            Rnamewordcount=len(' '.join(n))
            #print(Rnamewordcount)
            for word in n:
                Rwordlengths.append(len(word))
                if(len(Rwordlengths)<3):
                    Rwordlengths.append(0)
                    Rwordlengths.append(0)
                    #print(Rwordlengths)
        
       # else:
        #    Rwordlen=0
         #   Rnamewordcount=0
    #features.append(Rwordlengths[0])
    #features.append(Rwordlengths[1])
    #features.append(Rwordlengths[2])
    features.append(Rwordlen)
    features.append(RwordsC)
    features.append(Rnamewordcount)
    #features.append(0)
    #features.append(0)
    
    #print(features)
    
    return features
    
    


#doextraction(glob_text)


#if __name__ == '__main__':
    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
def trainingpart():
    x,y=doextraction(glob_text)
    m=training(x,y)
    return m,x,y
def testingpart(text):
    red,true=redact(text)
    x_test=redact_features(red)
    return x_test,true
def main():
    z=[]
    Out=""
    Mod,w,q=trainingpart()
    for thefile in glob.glob(glob_text2):
        #print(thefile)
        with io.open(thefile, 'r', encoding='utf-8',errors='ignore') as fyl:
            text = fyl.read()
            #print(text)
            xtest,true_n=testingpart(text)
            for i in xtest:
                if (i==''):
                    xtest=[0,0,0]
            #Xtest.append(xtest)
            
            R,T=predict(Mod,xtest)  
            z.append(R)
            
    
    return Mod,xtest,R,z

a,b,c,d=main()
precision(d)

#if __name__ == '__main__':
    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
 #   Mod=trainingpart(sys.argv[-1])
