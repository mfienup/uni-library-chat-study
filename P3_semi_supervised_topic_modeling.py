""" File:  P3_semi_supervised_topic_modeling.py  

    Description:  Loads a previously created pre-processed chat corpus, then performs
    semi-supervised topic modeling utilizing CorEx and GuidedLDA.

    INPUT FILES:
    0) anchors.txt - anchor/seed words each on their own line
    
    Previously created preprocessed chat corpus from either:
    1) wholeChatsFilePOS_N_ADJ_V.txt -- preprocessing keeping nouns, adjectives, and verbs
    2) wholeChatsFilePOS_N_ADJ.txt -- preprocessing keeping nouns and adjectives
    3) wholeChatsFile.txt -- NO POS preprocessing so all parts of speech
    4) onlyQuestionsFile.txt -- Only initial question of chats

    OUTPUT FILES:
    1) "raw_" text (.txt) file listing topics with each word scored
    2) "LDA_" text (.txt) file containing only the text for the
       specified number of topics with the specified number of words per topic

    Acknowledgements:

    Here we are used the CorEx (Correlation Explanation) package available at GitHub:
    https://github.com/gregversteeg/corex_topic

    Here we are used the GuidedLDA package is available at GitHub:
    https://github.com/vi3k6i5/GuidedLDA
    NOTE:  We had difficulty installing GuidedLDA, but we were finally successful
    by following the work-around posted at:
    https://github.com/dex314/GuidedLDA_WorkAround

"""
import os.path
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


from P2_utility_functions import *
from P3_utility_functions import *

def main():
    print('Welcome to Phase 3 which runs the semi-supervised topic modeling techniques.',
          '\n\nYou should have first run Phase 1 to pre-process your chat data.',
          '\nIt would generate cleaned chat files varying the parts of speech or question-only.',
          '\nFiles generated are: wholeChatsFile.txt, wholeChatsFilePOS_N_ADJ_V.txt,',
          '\nwholeChatsFilePOS_N_ADJ.txt, and onlyQuestionsFile.txt.\n\n')
    print('\n\nYou could have also run Phase 2 to execute unsupervised topic modeling techniques.',
          '\nIt would generate files: possible_2_word_anchors.txt and possible_3_word_anchors.txt which',
          '\nyou might use to create a text-file (.txt) with anchors one per line.\n')

    prompt = "\nStep 1. Please input the pre-processed (.txt) file." + \
             '\n(For example: "wholeChatsFile.txt"):'
    fileName = getFileName(prompt)
    chats = readChatCorpusFile(fileName)

    prompt = "\nStep 2. Please input the anchors/seeds (.txt) file." + \
             '\n(For example: "anchors.txt"):'
    fileName = getFileName(prompt)
    anchorList = readAnchorsFile(fileName)

    modelDict = {'GuidedLDA':run_GuidedLDA,'CorEx':run_CorEx}

    n_topics = getPositiveInteger('\nStep 3. Please specify the number of topics. (suggested range 10-20)\n')
    n_words_per_topic = getPositiveInteger('\nStep 4. Please specify the number of words per topics. (suggested range 5-10)\n')

    for model in modelDict:
        print("="*35)
        print("\nPerforming", model,"topic modeling -- please wait it might take a couple minutes!")
        topicList = modelDict[model](chats, anchorList, n_topics, n_words_per_topic)
        averagePMI, averageLCP, averageNZ = calculateTopicCoherenceMetrics(chats, topicList)
        print("\nResults for",model," TC-PMI %3.3f, TC-LCP %3.3f, TC-NZ %3.3f:" % (averagePMI, averageLCP, averageNZ))
        for topic in topicList:
            print(topic)
        

       
main()
