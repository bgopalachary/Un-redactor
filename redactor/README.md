

Bhavana Gopalachary
Bhavana.Gopalachary-1@ou.edu
Project 2 : UnRedactor

In this project the main aim was to predict names from large amounts of unredacted text. The data set we took for this project was movie review files from IMDB. The data set was already divided in to training and testing . 
Function 1:
The first thing I did was to read the data using glob and io and put the text in a variable. Then I performed the sent_tokenize and word_tokenize on the text data and using the labels I got after chunking, I took all the names from the text and put them in a list. Using this list, I extracted 3 features for all the names. The features were- number of words in a file, number of names in the file and the length of the name. After extracting these features, all the features are appended in to a single variable that is returned by the get_entity function along with the names. The names are y_train and the features are x_train.
Function 2:
In the next function, the returned data x_train and y_train are appended and returned .
Function 3:
This next function is about fitting the data in to a model and training it. The model I used is the MultiNomial Naïve Bayes. Before fitting the data, I converted them in to Numpy array.
Function 4:
In this function, the new testing data is read from files using glob and io and then by using the same procedure as that of the first function names are extracted from this text. The extra thing being done for this data is the redaction of the names in the text files. The redacted data is returned.
Function 5:
In this function, the redacted data is taken and feature extraction is done on this. The same features that have been extracted for the unredacted data are extracted for this redacted data. Here since the data is redacted, instead of counting the length of the word, the number of the special character is counted. These extracted features are returned as x_test.
Function 6:
Now this returned x_test is sent in to the .predict function and the names for each file are predicted.
Function 7:
In this function, using the sklear metrics the precision is found for the predicted data and the original data.

Imported modules:
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge 
from sklearn.metrics import precision_recall_fscore_support
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

References:
scikit-learn.org
datascience.stackexchange.com
dataaspirant.com
stackoverflow.com
www.pythonforbeginners.com
discuss.analyticsvidhya.com
www.albertauyeung.com
www.w3schools.com
Command to run tests:
pipenv run pytest test_unredactor.py 


# redactor
