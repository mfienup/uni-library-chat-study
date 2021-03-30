""" File:  P2_unsupervised_topic_modeling.py  

    Description:  Loads a previously created preprocessed chat corpus, then performs
    topic modeling utilizing unsupervised techniques of:
    1) Latent Semantic Analysis (TF-IDF & LSA) using gensim
    2) probabilistic Latent Semantic Analysis (TF-IDF & pLSA) using gensim
    3) Latent Dirichlet Allocation (LDA) using scikit-learn.org (sklearn) LDA module
    4) Latent Dirichlet Allocation (LDA), PyMallet
    Here we are used the LDA implementation from GitHub PyMallet at:
    https://github.com/mimno/PyMallet
    The LDA code below is based on their lda_reference.py code written in Python
    The PyMallet project has an MIT License see below.
================================================================================
MIT License

Copyright (c) 2019 mimno

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===========================================================================
    INPUT FILES:  User inputs file to process
    Previously created preprocessed chat corpus from P1_preprocess_data.py either:
    1) wholeChatsFilePOS_N_ADJ_V.txt -- preprocessing keeping nouns, adjectives, and verbs
    2) wholeChatsFilePOS_N_ADJ.txt -- preprocessing keeping nouns and adjectives
    3) wholeChatsFile.txt -- NO POS preprocessing so all parts of speech
    4) onlyQuestionsFile.txt -- Only initial question of chats

    OUTPUT FILES for each of the 4 unsupervised topic modeling techniques:
    1) "raw_" text (.txt) file listing topics with each word scored
    2) "LDA_" text (.txt) file containing only the text for the
       specified number of topics with the specified number of words per topic

    OUTPUT FILES for to aid the semi-supervised topic modeling techniques of Phase 3:
    1) possible_2_word_anchors.txt most frequent 2-word occurrence across combined topics
       of all four unsupervised topic modeling techniques
    2) possible_3_word_anchors.txt most frequent 3-word occurrence across combined topics
       of all four unsupervised topic modeling techniques      
    
"""
import os.path
from pprint import pprint  # pretty-printer
from collections import defaultdict
from gensim import corpora
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from time import time

from P2_utility_functions import *

def main():
    # ask users to input the name of the csv file cleaned, make sure it contains the column of 'body'
    print('Welcome to Phase 2 which runs the unsupervised topic modeling techniques.',
          '\n\nYou should have first run Phase 1 to pre-process your chat data.',
          '\nIt would generate cleaned chat files varying the parts of speech or question-only.',
          '\nFiles generated are: wholeChatsFile.txt, wholeChatsFilePOS_N_ADJ_V.txt,',
          '\nwholeChatsFilePOS_N_ADJ.txt, and onlyQuestionsFile.txt.\n')

    prompt = "\nStep 1. Please input the pre-processed (.txt) file." + \
             '\n(For example: "wholeChatsFile.txt"):'
    fileName = getFileName(prompt)
    chats = readChatCorpusFile(fileName)

    modelDict = {'PyMallet LDA':run_PyMallet_LDA, 'LDA':runLDA,
                 'TF-IDF & LSA':run_TFIDF_LSA, 'TF-IDF & pLSA':run_TFIDF_pLSA}

    n_topics = getPositiveInteger('\nStep 2. Please specify the number of topics. (suggested range 10-20)\n')
    n_words_per_topic = getPositiveInteger('\nStep 3. Please specify the number of words per topics. (suggested range 5-10)\n')

    combinedTopicsAcrossAllTechniques = []
    for model in modelDict:
        print("="*35)
        print("\nPerforming", model,"topic modeling -- please wait it might take a couple minutes!")
        topicList = modelDict[model](chats, n_topics, n_words_per_topic)
        averagePMI, averageLCP, averageNZ = calculateTopicCoherenceMetrics(chats, topicList)
        print("\nResults for",model," TC-PMI %3.3f, TC-LCP %3.3f, TC-NZ %3.3f:" % (averagePMI, averageLCP, averageNZ))
        for topic in topicList:
            print(topic)
        combinedTopicsAcrossAllTechniques.extend(topicList)

    # generate files of possible anchors for semi-supervised topic modeling techniques
    coOccurrenceDict, triOccurrenceDict = generate_Co_and_Tri_occurrence_dictionary(combinedTopicsAcrossAllTechniques,n_words_per_topic)
    writeOccurrenceFile(2, coOccurrenceDict)
    writeOccurrenceFile(3, triOccurrenceDict)

def runLDA(documents,n_topics, n_words_per_topic, max_features=1000, stop_words='english'):
    """ Performs LDA topic modeling and return resulting topics as strings in topicList """
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=max_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    # Fit the LDA model
    lda_model = LatentDirichletAllocation(n_topics, max_iter=50, learning_method='online',
                                    learning_decay = 0.7,
                                    learning_offset=50.,
                                    random_state=0)
    lda_fit = lda_model.fit(tf)
    lda_output = lda_model.transform(tf)

    fileName = "LDA_"+"_"+str(n_topics)+"topics_"+str(n_words_per_topic)+"words.txt"
    topicList = write_file_top_words(lda_fit, tf_feature_names, n_words_per_topic, fileName)
    return topicList

def run_TFIDF_pLSA(documents,n_topics, n_words_per_topic, max_features=1000, stop_words='english'):
    """ Performs TF-IDF and pLSA topic modeling and return resulting topics as strings in topicList """
    # Vectorize raw documents to tf-idf matrix: 
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    tfidf = tfidf_vectorizer.fit_transform(documents)
    nmf = NMF(n_components=n_topics, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    fileName = "TFIDF_pLSA_"+str(n_topics)+"topics_"+str(n_words_per_topic)+"words.txt"
    topicList = write_file_top_words(nmf, tfidf_feature_names, n_words_per_topic, fileName)
    return topicList

def run_PyMallet_LDA(documents, n_topics, n_words_per_topic, fileNameCorpus=""):
    """ Performs PyMallet LDA topic modeling and return resulting topics as strings in topicList """
    vocabulary, word_topics = PyMallet_LDA(documents, n_topics)
    fileName = "PyMallet_LDA_"+fileNameCorpus+"_"+str(n_topics) \
               +"topics_"+str(n_words_per_topic)+"words.txt"

    topicList = write_PyMallet_LDA(vocabulary, word_topics, n_topics, n_words_per_topic, fileName)

    return topicList

def run_TFIDF_LSA(documents,n_topics, n_words_per_topic):
    """ Performs TF-IDF and LSA topic modeling and return resulting topics as strings in topicList """
    stoplist = set()  # preprocessing removed stop words already...
    dictionary, corpus = createCorpusDictionary(documents, stoplist)
    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model

    # Apply a transformation to a whole corpus 
    corpus_tfidf = tfidf[corpus]

    # Initialize an LSI transformation
    numberOfTopics = 300 # (recommended between 200-500)
    lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics=numberOfTopics)
    # create a double wrapper over chat corpus: bow -> tfidf -> fold-in-lsi
    corpus_lsi = lsi[corpus_tfidf]
    fileName = "TFIDF_LSA_"+str(n_topics)+"topics_"+str(n_words_per_topic)+"words.txt"
    topicList = write_LSA(lsi, n_topics, n_words_per_topic, fileName)
    return topicList
    
def generate_Co_and_Tri_occurrence_dictionary(combinedTopicsAcrossAllTechniques,n_words_per_topic):
    """ To aid the semi-supervised topic modeling techniques of Phase 3, determines the
        co-occurrences (2-words) and tri-occurrences across combined topics of all four
        unsupervised topic modeling techniques.
    """
    coOccurrenceDict = {}
    triOccurrenceDict = {}
    wordFreqDict = {}
    wordFreqDict2 = {}
    windowSize = n_words_per_topic
    stopWordDict = {}  #stop words previously removed
    combinedTopicsFile = open("combinedTopicsFile.txt", 'w')
    for topic in combinedTopicsAcrossAllTechniques:
        document = topic.split()
        tallyTriOccurrencesInWindow(document, windowSize, triOccurrenceDict, wordFreqDict, stopWordDict)
        tallyCoOccurrencesInWindow(document, windowSize, coOccurrenceDict, wordFreqDict2, stopWordDict)
        combinedTopicsFile.write(topic+"\n")
    combinedTopicsFile.close()
    return coOccurrenceDict, triOccurrenceDict

def writeOccurrenceFile(occurrenceSize, occurrenceDict):
    """ Called twice to generate two files to aid Phase 3 semi-supervised topic modeling:
        1) possible_2_word_anchors.txt most frequent 2-word occurrence across combined
           topics of all four unsupervised topic modeling techniques, and
        2) possible_3_word_anchors.txt most frequent 3-word occurrence across combined
           topics of all four unsupervised topic modeling techniques.
    """
    occurrencesFile = open("possible_"+str(occurrenceSize)+"_word_anchors.txt", 'w')
    occurrencesFile.write("Possible "+str(occurrenceSize)+" word anchors for semi-supervised topic modeling.\n")
    occurrencesFile.write("Found from most frequently occuring "+ str(occurrenceSize)+ "-word occurrences from\n" +
                          "all topics found by supervised topic modeling techniques:\n" +
                          "LDA, PyMallet_LDA, pLSA, and LSA\n\n")
    countList = []
    for wordTuple, count in occurrenceDict.items():
        countList.append((count, wordTuple))
    countList.sort()
    countList.reverse()
    numberToSee = min(len(countList), 50)
    for index in range(numberToSee):
        count, wordTuple = countList[index]
        occurrencesFile.write("tuple count: %d  words %s\n" % (count, str(wordTuple)))
    occurrencesFile.close()
    

main()   

