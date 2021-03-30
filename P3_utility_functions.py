""" File:  P3_utility_functions.py  

    Description:  Utility functions to performs
    semi-supervised topic modeling utilizing CorEx and GuidedLDA.

    Acknowledgements:

    Here we are used the CorEx (Correlation Explanation) package available at GitHub:
    https://github.com/gregversteeg/corex_topic

    Here we are used the GuidedLDA package is available at GitHub:
    https://github.com/vi3k6i5/GuidedLDA
    NOTE:  We had difficulty installing GuidedLDA, but we were finally successful
    by following the work-around posted at:
    https://github.com/dex314/GuidedLDA_WorkAround

"""import os.path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from corextopic import corextopic as ct
import pandas as pd
import nltk
from time import time

import re, sys, random, math
import numpy as np
from lda import guidedlda as glda
from lda import glda_datasets as gldad

from collections import Counter
from timeit import default_timer as timer


def readAnchorsFile(fileName):
    """ Reads anchor/seeds from fileName and returns list-of-lists anchorList """
    anchorList = []
    anchorFile = open(fileName, 'r')
    for line in anchorFile:
        wordList = line.strip().split()
        if len(wordList) > 0:
            anchorList.append(wordList)
    anchorFile.close()

    return anchorList

def run_GuidedLDA(chats, anchorList, n_topics, n_words_per_topic,SEED_CONFIDENCE=0.75):
    """ Perform GuidedLDA on corpus from chats using anchorList.
        Returns topics as strings in topicList.
    """
    word2id = {}
    docs = []
    id2word = {}
    wordList = []
    wordId = 0
    for documentLine in chats:
        newDoc = ""
        for word in documentLine.split():
            if word not in word2id:
                word2id[word] = wordId
                id2word[wordId] = word
                wordList.append(word)
                wordId += 1
            newDoc += word + " "
        if len(newDoc) > 0:
            docs.append(newDoc)
    numDocs = len(docs)
    numWords = len(word2id)
    vocab = tuple(wordList)

    X = np.ndarray(shape=(numDocs, numWords), dtype=int)

    word_counts = Counter()
    documents = []
    word_topics = {}
    topic_totals = np.zeros(n_topics)

    for docIndex, docLine in enumerate(docs):
        
        for word in docLine.strip().split():
            wordId = word2id[word]
            X[docIndex][wordId] += 1

    seed_topic_list = anchorList
    model = glda.GuidedLDA(n_topics=n_topics, n_iter=100,
                           random_state=7, refresh=20)
    seed_topics = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            seed_topics[word2id[word]] = t_id

    model.fit(X, seed_topics=seed_topics, seed_confidence=SEED_CONFIDENCE)

    # Display and write to file the results of CorEx with no anchors
    fileName = "GuidedLDA_seeds_"+str(len(seed_topic_list))+"_confidence_"+ \
               str(SEED_CONFIDENCE)+"_"+str(n_topics) +"topics_"+str(n_words_per_topic)+"words.txt"
    outputFile = open(fileName, 'w')
    outputFile.write("File: " + fileName +"\n\n")
    topicList = []
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_words_per_topic+1):-1]
        topicStr = '{}'.format(' '.join(topic_words))
        topicList.append(topicStr)     
        outputFile.write(topicStr+"\n")
    outputFile.close()
    return topicList

def run_CorEx(documents, anchorList, n_topics, n_words_per_topic):
    """ Performs CorEx on corpus documents using anchorList.
        Returns topics as strings in topicList.
    """
    # CorEx uses an TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_df=.5, min_df=10, max_features=None,
    ##    ngram_range=(1, 2),  for bi-grams
    ##    ngram_range=(1,3),   for bi-grams and tri-grams
        ngram_range=(1,1),     # for no bi-grams or tri-grams
        norm=None,
        binary=True,
        use_idf=False,
        sublinear_tf=False
    )

    # Fit chat corpus to TF-IDF vectorization
    vectorizer = vectorizer.fit(documents)
    tfidf = vectorizer.transform(documents)
    vocab = vectorizer.get_feature_names()

    # Apply CorEx with no anchors for a comparison
    anchors = []
    model = ct.Corex(n_hidden=n_topics, seed=42) # n_hidden specifies the # of topics
    model = model.fit(tfidf, words=vocab)

    # Display and write to file the results of CorEx with no anchors
    fileName = "CorEx_no_anchors_"+str(n_topics)+"topoics_"+str(n_words_per_topic)+"words.txt"
    outputFile = open(fileName, 'w')
    outputFile.write("File: " + fileName +"\n\n")

    print("\nCorEx Topics with no anchors:")
    for i, topic_ngrams in enumerate(model.get_topics(n_words=n_words_per_topic)):
        topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
        print("Topic #{}: {}".format(i+1, ", ".join(topic_ngrams)))
        outputFile.write("{}".format(" ".join(topic_ngrams))+"\n")
    outputFile.close()

    ## remove anchor words that are not in the chat corpus
    anchors = [
        [a for a in topic if a in vocab]
        for topic in anchorList
    ]

    model = ct.Corex(n_hidden=n_topics, seed=42)
    model = model.fit(
        tfidf,
        words=vocab,
        anchors=anchors, # Pass the anchors in here
        anchor_strength=3 # Tell the model how much it should rely on the anchors
    )

    # Display and write to file the results of CorEx with no anchors
    fileName = "CorEx_anchors_"+str(len(anchors))+"_"+str(n_topics) \
               +"topoics_"+str(n_words_per_topic)+"words.txt"
    outputFile = open(fileName, 'w')
    outputFile.write("File: " + fileName +"\n\n")
    topicList = []
    print("\nCorEx Topics with anchors:")
    for i, topic_ngrams in enumerate(model.get_topics(n_words=n_words_per_topic)):
        topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
        topicList.append(" ".join(topic_ngrams))
        print("Topic #{}: {}".format(i+1, ", ".join(topic_ngrams)))
        outputFile.write("{}".format(" ".join(topic_ngrams))+"\n")
    outputFile.close()    
    return topicList


