""" File:  P2_utility_functions.py
    Unility functions for Phase 2 which performs the topic modeling.

    Latent Dirichlet Allocation (LDA), PyMallet
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
"""
from pprint import pprint  # pretty-printer
from collections import defaultdict
from gensim import corpora
from six import iteritems

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import os.path

from sklearn.pipeline import Pipeline
from time import time


def getStopWords(stopWordFileName):
    """Reads stop-words text file which is assumed to have one word per line.
       Returns stopWordDict.
    """
    stopWordDict = {}
    stopWordFile = open(stopWordFileName, 'r')

    for line in stopWordFile:
        word = line.strip().lower()
        stopWordDict[word] = None
    stopWordSet = set(stopWordDict)
        
    return stopWordDict, stopWordSet

def getPositiveInteger(prompt):
    """Prompts the user for a valid positive integer which it returns.
    """
    while True:
        inputStr = input(prompt+" ")
        try:
            intValue = int(inputStr)
            if intValue <= 0:
                print("Please enter a positive integer.")
                raise ValueError("positive integer only")
            return intValue
        except:
            print("Invalid positive integer")

def getFileName(prompt):
    """Prompts the user for a valid file which it returns.
    """
    while True:
        fileName = input(prompt+" ")
        if os.path.exists(fileName):
            return fileName
        else:
            print("File not found! Make sure that the file is inside this directory.")

# used in LDA sklearn
def readChatCorpusFile(chatFileName):
    """ Read specified chat corpus file (which should be a preprocessed
        text file with one chat per line) and returns the documents list
        where each chat being a string in the list.
    """
    documentsFile = open(chatFileName, 'r')

    documentsList = []
    for documentLine in documentsFile:
        documentLine = documentLine.lower()
        if len(documentLine) > 0:
            documentsList.append(documentLine)
    #print("len(documents)",len(documentsList))
    return documentsList

def print_top_words(model, feature_names, n_top_words):
    """ Displays the specified top topics and top words to screen"""
    for topic_idx, topic in enumerate(model.components_):
        
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        message += "\n "
        message += " ".join([feature_names[i]+" ("+str(model.components_[topic_idx][i])+")\n"
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def write_file_top_words(model, feature_names, n_top_words, fileName):
    """ Writes the specified top topics and top words to the specified fileName"""
    outputFile = open("raw_"+fileName, 'w')
    outputFileTopics = open(fileName, 'w')
    outputFile.write("File: "+"raw_"+fileName+"\n\n")
    outputFileTopics.write("File: "+fileName+"\n\n")
    topicList = []
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        topicStr = " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        topicList.append(topicStr)
        outputFileTopics.write(topicStr+"\n")

        message += topicStr + "\n "
        message += " ".join([feature_names[i]+" ("+str(model.components_[topic_idx][i])+")\n"
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        outputFile.write(message+"\n")

    outputFile.close()
    outputFileTopics.close()
    return topicList

# used in LSA with gensim
def write_LSA(model_lsi, n_topics, n_words_per_topic, fileName):
    """ Writes the specified top topics and top words to the specified fileName"""
    topicList = model_lsi.print_topics(n_topics)
    ##print("topicList",topicList)
    corpus_tfidf_and_lsa_fileName = "raw_" + fileName
    corpus_tfidf_and_lsa_file = open(corpus_tfidf_and_lsa_fileName, 'w')
    outputFile = open(fileName,'w')
    corpus_tfidf_and_lsa_file.write("File: " + corpus_tfidf_and_lsa_fileName + '\n\n')
    outputFile.write("File: " + fileName + '\n\n')
    listOfTopics = []
    for topicIndex in range(n_topics):
        corpus_tfidf_and_lsa_file.write(str(topicList[topicIndex])+'\n')
        line = str(topicList[topicIndex])
        topicString = ""
        startIndex = 0
        for count in range(n_words_per_topic):
            wordStart = line.find('*"', startIndex) + 2
            wordEnd = line.find('"', wordStart) - 1
            topicString += line[wordStart:wordEnd+1] + " "
            startIndex = wordEnd + 1
        outputFile.write(topicString+"\n")
        listOfTopics.append(topicString)

    outputFile.close()
    corpus_tfidf_and_lsa_file.close()
    return listOfTopics


# create a vector stream to avoid loading the whole vector into memory at one time
class MyCorpus(object):
    def __init__(self, documentsList, dictionary):
        self._docsList = documentsList
        self.myDictionary = dictionary
        
    def __iter__(self):
#        for line in open(FILE_NAME_OF_CORPUS+'_lemmatized.txt'):
        for line in self._docsList:
            # assume there's one document per line, tokens separated by whitespace
            yield self.myDictionary.doc2bow(line.lower().split())


def createCorpusDictionary(documentsList, stoplist):
    # collect statistics about all tokens, i.e., words
    dictionary = corpora.Dictionary(line.lower().split() for line in documentsList)
    # remove stop words and words that appear only once
##    stop_ids = [
##        dictionary.token2id[stopword]
##        for stopword in stoplist
##        if stopword in dictionary.token2id
##    ]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
##    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.filter_tokens(once_ids)  # remove words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed

    #dictionary.save(FILE_NAME_OF_CORPUS+'.dict')  # store the dictionary, for future reference
    corpus_memory_friendly = MyCorpus(documentsList,dictionary)  # doesn't load the corpus into memory!

    #corpora.MmCorpus.serialize(FILE_NAME_OF_CORPUS+'.mm', corpus_memory_friendly)
    return dictionary, corpus_memory_friendly

## functions used in PyMallet_LDA
""" 
    Here we are used the LDA implementation from GitHub PyMallet at:
    https://github.com/mimno/PyMallet
    The LDA code below is based on their lda_reference.py code written in Python
    The PyMallet project has an MIT License see below.

    INPUT FILES:
    Previously created preprocessed chat corpus from either:
    1) wholeChatsFilePOS_N_ADJ_V.csv -- preprocessing keeping nouns, adjectives, and verbs
    2) wholeChatsFilePOS_N_ADJ.csv -- preprocessing keeping nouns and adjectives
    3) wholeChatsFile.csv -- NO POS preprocessing so all parts of speech
    4) onlyQuestionsFile.csv -- Only initial question of chats

    OUTPUT FILES:
    1) "raw_" text (.txt) file listing topics with each word scored
    2) "PyMallet_LDA_" text (.txt) file containing only the text for the
       specified number of topics with the specified number of words per topic

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

"""

import re, sys, random, math
import numpy as np
from collections import Counter
from timeit import default_timer as timer

from time import time

def sample(documents, vocabulary_size, word_topics, topic_totals, word_counts, num_iterations, n_topics, doc_smoothing = 0.5, word_smoothing = 0.01):
    smoothing_times_vocab_size = word_smoothing * vocabulary_size

    word_pattern = re.compile("\w[\w\-\']*\w|\w")

    for iteration in range(num_iterations):
        
        for document in documents:
            
            doc_topic_counts = document["topic_counts"]
            token_topics = document["token_topics"]
            doc_length = len(token_topics)
            for token_topic in token_topics:
                
                w = token_topic["word"]
                old_topic = token_topic["topic"]
                word_topic_counts = word_topics[w]
                
                ## erase the effect of this token
                word_topic_counts[old_topic] -= 1
                topic_totals[old_topic] -= 1
                doc_topic_counts[old_topic] -= 1
                
                ###
                ### SAMPLING DISTRIBUTION
                ###
                
                ## Does this topic occur often in the document?
                topic_probs = (doc_topic_counts + doc_smoothing) / (doc_length + n_topics * doc_smoothing)
                ## Does this word occur often in the topic?
                topic_probs *= (word_topic_counts + word_smoothing) / (topic_totals + smoothing_times_vocab_size)
                
                ## sample from an array that doesn't sum to 1.0
                sample = random.uniform(0, np.sum(topic_probs))
                
                new_topic = 0
                while sample > topic_probs[new_topic]:
                    sample -= topic_probs[new_topic]
                    new_topic += 1
                
                ## add back in the effect of this token
                word_topic_counts[new_topic] += 1
                topic_totals[new_topic] += 1
                doc_topic_counts[new_topic] += 1
                
                token_topic["topic"] = new_topic               

def entropy(p):
    ## make sure the vector is a valid probability distribution
    p = p / np.sum(p)
    
    result = 0.0
    for x in p:
        if x > 0.0:
            result += -x * math.log2(x)
            
    return result

def print_topic(topic):
    sorted_words = sorted(vocabulary, key=lambda w: word_topics[w][topic], reverse=True)
    
    for i in range(20):
        w = sorted_words[i]
        print("{}\t{}".format(word_topics[w][topic], w))

def print_all_topics():
    for topic in range(NUMBER_OF_TOPICS_PRINTED):
        sorted_words = sorted(vocabulary, key=lambda w: word_topics[w][topic], reverse=True)
        print(" ".join(sorted_words[:20]))


def PyMallet_LDA(docs, n_topics, stoplist = set()):
    word_pattern = re.compile("\w[\w\-\']*\w|\w")
    word_counts = Counter()

    documents = []
    word_topics = {}
    topic_totals = np.zeros(n_topics)


    for line in docs:
        #line = line.lower()
        
        tokens = word_pattern.findall(line)
        
        ## remove stopwords, short words, and upper-cased words
        tokens = [w for w in tokens if not w in stoplist and len(w) >= 3 and not w[0].isupper()]
        word_counts.update(tokens)
        
        doc_topic_counts = np.zeros(n_topics)
        token_topics = []
        
        for w in tokens:
            
            ## Generate a topic randomly
            topic = random.randrange(n_topics)
            token_topics.append({ "word": w, "topic": topic })
            
            ## If we haven't seen this word before, initialize it
            if not w in word_topics:
                word_topics[w] = np.zeros(n_topics)
            
            ## Update counts: 
            word_topics[w][topic] += 1
            topic_totals[topic] += 1
            doc_topic_counts[topic] += 1
        
        documents.append({ "original": line, "token_topics": token_topics, "topic_counts": doc_topic_counts })

    ## Now that we're done reading from disk, we can count the total
    ##  number of words.
    vocabulary = list(word_counts.keys())
    vocabulary_size = len(vocabulary)

    num_iterations = 100
    sample(documents, vocabulary_size, word_topics, topic_totals, word_counts, num_iterations, n_topics, doc_smoothing = 0.5, word_smoothing = 0.01)

    return vocabulary, word_topics

def write_PyMallet_LDA(vocabulary, word_topics, n_topics, n_words_per_topic, fileName):
    """ Writes the results of PyMallet LDA to files and returns the resulting topics as
        stings in topicList.
    """
    outputFile = open(fileName, 'w')
    outputFile.write("File: " + fileName +"\n\n")
    rawFileName = "raw_"+fileName
    outputFileRaw = open(rawFileName, 'w')
    outputFileRaw.write("File: " + rawFileName +"\n\n")
    topicList = []
    for topic in range(n_topics):
        sorted_words = sorted(vocabulary, key=lambda w: word_topics[w][topic], reverse=True)
        topicStr = " ".join(sorted_words[:n_words_per_topic])
        topicList.append(topicStr)
        outputFile.write(topicStr+"\n")
        outputFileRaw.write(topicStr+"\n")
        #print(topicStr)
        for i in range(n_words_per_topic):
            w = sorted_words[i]
            #print("{}\t{}".format(word_topics[w][topic], w))
            outputFileRaw.write("{}\t{}".format(word_topics[w][topic], w) +"\n")
        
    outputFile.close()
    outputFileRaw.close()
    return topicList

## functions used in the topic coherence metric calculations
import math

EPSILON = 0.000000001

def calculateTopicCoherenceMetrics(documentsList, topicsList, stopWordDict = {}):
    """ Calculates and returns the topic coherence metrics: averagePMI, averageLCP, averageNZ
        for the set of topics in topicsList and the reference corpus in documentsList
    """
    outputFileName = "TC_metrics_.txt"

    coOccurrenceDict = {}
    wordDict = {}
    topicsList, topicsCoOccurrenceList, coOccurrenceDict, wordDict = findcoOoccurrencesAndWordsInTopics(topicsList)

    numberOfTopics = len(topicsList)
    
    docCount = tallycoOoccurrencesAndWordsInDocs(documentsList, coOccurrenceDict, wordDict)

    makeProbabilities(docCount, coOccurrenceDict, wordDict)

    outputFile = open(outputFileName, 'w')

    outputFile.write("File: "+outputFileName+"\n\n")
    
    sumPMI = 0.0
    sumLCP = 0.0
    sumNZ = 0
    index = 0
    for topicCoOccurrence in topicsCoOccurrenceList:
        topicPMI = calculateTopicPMI(topicCoOccurrence, coOccurrenceDict, wordDict)
        topicLCP = calculateTopicLCP(topicCoOccurrence, coOccurrenceDict, wordDict)
        topicNZ = calculateTopicNZ(topicCoOccurrence, coOccurrenceDict)
        outputFile.write(topicsList[index]+"\n")
        outputFile.write("PMI = %.3f  " % (topicPMI))
        outputFile.write("LCP = %.3f  " % (topicLCP))
        outputFile.write("NZ = %d\n" % (topicNZ))
        sumPMI += topicPMI
        sumLCP += topicLCP
        sumNZ += topicNZ
        index += 1
    averagePMI = sumPMI/numberOfTopics
    averageLCP = sumLCP/numberOfTopics
    averageNZ = sumNZ/numberOfTopics
    outputFile.write("\nAverage PMI of all topics: %.3f\n" % (averagePMI))
    outputFile.write("\nAverage LCP of all topics: %.3f\n" % (averageLCP))
    outputFile.write("\nAverage NZ of all topics: %.3f\n" % (averageNZ))
    outputFile.close()
    return averagePMI, averageLCP, averageNZ

def makeProbabilities(docCount, coOccurrenceDict, wordDict):
    """ Converses the raw counts in the coOccurrenceDict and wordDict into probabilities."""
    for coOccurrence in coOccurrenceDict:
        coOccurrenceDict[coOccurrence] /= float(docCount)
    for word in wordDict:
        wordDict[word] /= float(docCount)

def calculateTopicPMI(topicCoOccurrenceList, coOccurrenceDict, wordDict):
    """ Calculates and returns a topic's total PMI. """ 
    sumPMI = 0.0
    for topicCoOccurrence in topicCoOccurrenceList:
        sumPMI += calculatePMI(topicCoOccurrence, coOccurrenceDict, wordDict)
    return sumPMI/len(topicCoOccurrenceList)

def calculateTopicLCP(topicCoOccurrenceList, coOccurrenceDict, wordDict):
    """ Calculates and returns a topic's total LCP. """ 
    sumLCP = 0.0
    for topicCoOccurrence in topicCoOccurrenceList:
        firstWord, secondWord = topicCoOccurrence
        sumLCP += calculateLCP(firstWord, topicCoOccurrence, coOccurrenceDict, wordDict)
        sumLCP += calculateLCP(secondWord, topicCoOccurrence, coOccurrenceDict, wordDict)
    return sumLCP/(2*len(topicCoOccurrenceList))

def calculateTopicNZ(topicCoOccurrenceList, coOccurrenceDict):
    """ Calculates and returns a topic's total NZ. """ 
    sumNZ = 0
    for topicCoOccurrence in topicCoOccurrenceList:
        if coOccurrenceDict[topicCoOccurrence] == 0.0:
            sumNZ += 1
    return sumNZ

def calculatePMI(topicCoOccurrence, coOccurrenceDict, wordDict):
    """ Calculates and returns the PMI for a pair of words in the topicCoOccurrence tuple. """
    wordI, wordJ = topicCoOccurrence
    PMI = math.log((coOccurrenceDict[topicCoOccurrence]+EPSILON)/(wordDict[wordI]*wordDict[wordJ]),10)
    return PMI
        
        
def calculateLCP(word, topicCoOccurrence, coOccurrenceDict, wordDict):
    """ Calculates and returns the LCP for a word in the pair of words in the topicCoOccurrence tuple. """
    LCP = math.log((coOccurrenceDict[topicCoOccurrence]+EPSILON)/(wordDict[word]),10)
    return LCP
                
def tallycoOoccurrencesAndWordsInDocs(documentsList, coOccurrenceDict, wordDict):
    """ Tallys across all the documents in documentsList the word pair co-occurrences in coOccurrenceDict, and
        individual words in wordDict."""
    docCount = 0
    for document in documentsList:
        emptyDoc = tallyCoOccurrencesInDoc(document, coOccurrenceDict, wordDict)
        if not emptyDoc:
            docCount += 1
    return docCount

def tallyCoOccurrencesInDoc(document, coOccurrenceDict, wordDict):
    """ Tallys for an individual document the word pair co-occurrences in coOccurrenceDict, and
        individual words in wordDict."""
    docCoOccurrenceDict = {}
    docWordDict = {}
    
    wordList = document.strip().split()
    if len(wordList) == 0:
        return True   # empty document
    
    # eliminate duplicate words by converting to a set and back
    wordSet = set(wordList)
    wordList = list(wordSet)

    wordList.sort()
    for first in range(len(wordList)):
        if wordList[first] in wordDict:
            wordDict[wordList[first]] += 1
        for second in range(first+1,len(wordList)):
            coOccurrenceTuple = (wordList[first], wordList[second])
            if coOccurrenceTuple in coOccurrenceDict:
                coOccurrenceDict[coOccurrenceTuple] += 1
    return False   # not empty document

def findcoOoccurrencesAndWordsInTopics(topicsList):
    """ Processes the topics file and returns:
        topicsList - list of strings with one whole topic as a string,
        topicsCoOccurrenceList - a list-of-lists with the inner-list being the list word pairs as tuples within a topic,
        coOccurrenceDict - keys are tuple of word pairs that co-occur in the topics with their associated values of 0,
        wordDict - keys are words that occur in the topics with their associate values of 0."""

    topicsCoOccurrenceList = []
    coOccurrenceDict = {}
    wordDict = {}
    topicTupleList = []
    for line in topicsList:
        topicTupleList = []
        wordList = line.strip().split()
        wordList.sort()
        for first in range(len(wordList)):
            wordDict[wordList[first]] = 0
            for second in range(first+1,len(wordList)):
                coOccurrenceTuple = (wordList[first], wordList[second])
                coOccurrenceDict[coOccurrenceTuple] = 0
                topicTupleList.append(coOccurrenceTuple)
        topicsCoOccurrenceList.append(topicTupleList)
    return topicsList, topicsCoOccurrenceList,coOccurrenceDict, wordDict

def tallyTriOccurrencesInWindow(document, windowSize, triOccurrenceDict, wordFreqDict, stopWordDict):
    """ Tally the tri-occurrences of non-stop words in all documents of a given window size. """
    wordList = document

    initialChuckSize = min(len(wordList), windowSize)

    # process initial window size or whole line if it is smaller than window size
    for first in range(initialChuckSize-2):
        if wordList[first] in wordFreqDict:
            wordFreqDict[wordList[first]] += 1
        else:
            wordFreqDict[wordList[first]] = 1
            
        for second in range(first+1,initialChuckSize-1):          
            for third in range(second+1,initialChuckSize):          
                if wordList[first] != wordList[second] and \
                   wordList[first] != wordList[third] and \
                   wordList[second] != wordList[third] and \
                   wordList[first] not in stopWordDict and \
                   wordList[second] not in stopWordDict and \
                   wordList[third] not in stopWordDict:
                    words = [wordList[first],wordList[second],wordList[third]]
                    words.sort()
                    triOccurrenceTuple = (words[0], words[1], words[2])
                    if triOccurrenceTuple in triOccurrenceDict:
                        triOccurrenceDict[triOccurrenceTuple] += 1
                    else:
                        triOccurrenceDict[triOccurrenceTuple] = 1

    # slide the window down the whole length of the line
    for nextWordIndex in range(windowSize, len(wordList)):
        if wordList[nextWordIndex] in wordFreqDict:
            wordFreqDict[wordList[nextWordIndex]] += 1
        else:
            wordFreqDict[wordList[nextWordIndex]] = 1
        for second in range(nextWordIndex -1, nextWordIndex-windowSize+2, -1):
            for third in range(second-1,nextWordIndex-windowSize+1, -1):          
                if wordList[nextWordIndex] != wordList[second] and \
                   wordList[nextWordIndex] != wordList[third] and \
                   wordList[second] != wordList[third] and \
                   wordList[nextWordIndex] not in stopWordDict and \
                   wordList[second] not in stopWordDict and \
                   wordList[third] not in stopWordDict:
                    words = [wordList[nextWordIndex],wordList[second],wordList[third]]
                    words.sort()
                    triOccurrenceTuple = (words[0], words[1], words[2])
                    if triOccurrenceTuple in triOccurrenceDict:
                        triOccurrenceDict[triOccurrenceTuple] += 1
                    else:
                        triOccurrenceDict[triOccurrenceTuple] = 1

def tallyCoOccurrencesInWindow(document, windowSize, coOccurrenceDict, wordFreqDict, stopWordDict):
    """ Tally the co-occurrences of non-stop words in all documents of a given window size. """
    wordList = document

    initialChuckSize = min(len(wordList), windowSize)

    # process initial window size or whole line if it is smaller than window size
    for first in range(initialChuckSize):
        if wordList[first] in wordFreqDict:
            wordFreqDict[wordList[first]] += 1
        else:
            wordFreqDict[wordList[first]] = 1
            
        for second in range(first+1,initialChuckSize):          
            if wordList[first] != wordList[second] and \
               wordList[first] not in stopWordDict and \
               wordList[second] not in stopWordDict:
                if wordList[first] < wordList[second]:
                    coOccurrenceTuple = (wordList[first], wordList[second])
                elif wordList[first] > wordList[second]:
                    coOccurrenceTuple = (wordList[second], wordList[first])
                if coOccurrenceTuple in coOccurrenceDict:
                    coOccurrenceDict[coOccurrenceTuple] += 1
                else:
                    coOccurrenceDict[coOccurrenceTuple] = 1

    # slide the window down the whole length of the line
    for nextWordIndex in range(windowSize, len(wordList)):
        if wordList[nextWordIndex] in wordFreqDict:
            wordFreqDict[wordList[nextWordIndex]] += 1
        else:
            wordFreqDict[wordList[nextWordIndex]] = 1
        for otherWordIndex in range(nextWordIndex-windowSize+1, nextWordIndex):
            if wordList[nextWordIndex] != wordList[otherWordIndex] and \
               wordList[nextWordIndex] not in stopWordDict and \
               wordList[otherWordIndex] not in stopWordDict:
                if wordList[nextWordIndex] < wordList[otherWordIndex]:
                    coOccurrenceTuple = (wordList[nextWordIndex], wordList[otherWordIndex])
                elif wordList[nextWordIndex] > wordList[otherWordIndex]:
                    coOccurrenceTuple = (wordList[otherWordIndex], wordList[nextWordIndex])
                if coOccurrenceTuple in coOccurrenceDict:
                    coOccurrenceDict[coOccurrenceTuple] += 1
                else:
                    coOccurrenceDict[coOccurrenceTuple] = 1

