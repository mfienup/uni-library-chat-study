""" File:  P1_preprocess_data.py
    Description:  Takes as input raw chat data .csv file from the LibChat keeping only the columns:
    Timestamp, Duration (seconds), Initial Question, Message Count, and Transcript

    Additionally the chat text data is "cleaned" by: 
    1) removing timestamps, 
    2) removing chat patron and librarian identifiers, 
    3) removing http tags (e.g., URLs), 
    4) removing non-ASCII characters,
    5) removing stopwords, and 
    6) lemmatized words using nltk.WordNetLemmatizer() 

    Four data-set versions of the “cleaned” chat transcripts were prepared:
    1) "onlyQuestionsFile.txt" - Questions only: consists of only the initial question asked by
        the library patron in each chat transcript
    2) "wholeChatsFile.txt" - Whole chats: consists of the whole cleaned chat transcripts
    3) "wholeChatsFilePOS_N_ADJ.txt" - Whole chats with POS (Noun and Adjective): consists of only
       the nouns and adjectives parts-of-speech (POS) from the whole cleaned chat transcripts
    4) "wholeChatsFilePOS_N_ADJ_V.txt" - Whole chats with POS (Noun, Adjective, and Verb): consists
       of only the nouns, adjectives, and verbs parts-of-speech (POS) from the whole cleaned chat transcripts
    The goal of the first two data sets was to see if looking at only the initial question in the
    chats was better than the whole chats. The goal of the last two data sets was to see if varying
    the parts-of-speech retained had any effect on the topic modeling analyses. 

    Takes as input raw chat data .csv file and produces a list-of-lists called transcriptDialogList with a format:
    [[<excel index int>, "Initial question string", [Transcript split by chat responses which including initial
    question]], ...]. This transcriptDialogList is used to write two text files for each of the four
    data-set versions .  Each chat dialog is used to produce one line in the two text files:
    1) the .csv file is formated with one chat per line formatted as:
       chat line # in original .csv, cleaned and pre-processed text of the chat, and
    2) the .txt file is cleaned and pre-processed text of the chat

"""
import nltk

from P1_utility_functions import *

def main():
    print('Welcome to Phase 1 of the chat analysis which pre-processes a raw chat data .csv file',
          '\nfrom the LibChat keeping only the 5 columns (with column-headings):',
          '\nTimestamp, Duration (seconds), Initial Question, Message Count, and Transcript.',
          '\n\nRunning Phase 1 to pre-process your raw chat data (.csv) will generate four cleaned chat',
          '\nfiles varying the parts of speech or question-only.',
          '\n1) "onlyQuestionsFile.txt" - consists of only the initial questions asked by the library patrons',
          '\n2) "wholeChatsFile.txt" - consists of the whole cleaned chat transcripts',
          '\n3) "wholeChatsFilePOS_N_ADJ.txt" - consists of only the nouns and adjectives parts-of-speech (POS)',
          '\n4) "wholeChatsFilePOS_N_ADJ_V.txt" - consists of only the nouns, adjectives, and verbs parts-of-speech\n')

    prompt = "\nStep 1. Please input the raw LibChat (.csv) file." + \
             '\n(For example: "chatFile.csv"):'
    inputCSVFileName = getFileName(prompt)

    prompt = "\nStep 2. Please input the stop words (.txt) file." + \
             '\n(For example: "stop_words.txt"):'
    stopWordFileName = getFileName(prompt)

    print("\n\nWARNING:  Depending on the size of your chat data file.  This step might take several minutes.")

    POS_list = ['n','a','v','r']  # n - noun and a - adjective other possibilities: v -verb, r - adverb, 'other'

    stopWordsDict = getStopWords(stopWordFileName)
    transcriptList = readRawChats(inputCSVFileName)

    initialQuestionCount = 0
    transIndex = 2  # Assumes Excel .cvs had a column-header in line 1
    transcriptDialogList = []
    for trans in transcriptList:
        transDialogList = generateTranscriptDialogList(trans)
        initialQuestion = findInitialQuestion(trans, transIndex)
        if initialQuestion == None:
            initialQuestion = findInitialQuestionInDialog(transDialogList,transIndex)           
        else:
            initialQuestionCount+= 1
            
        transcriptDialogList.append([transIndex, initialQuestion, transDialogList])
        transIndex += 1

    print("Number of initial questions from Initial Question column of .csv:", initialQuestionCount)

    POS_list = ['n','a','v','r','other']  # n - noun and a - adjective other possibilities: v -verb, r - adverb, 'other'
    writeQuestionsOnlyToFile(transcriptDialogList, "onlyQuestionsFile", stopWordsDict, POS_list)

    writeWholeChatsToFile(transcriptDialogList, "wholeChatsFile", stopWordsDict, POS_list)

    POS_list = ['n','a']  # n - noun and a - adjective other possibilities: v -verb, r - adverb, 'other'
    writeWholeChatsToFile(transcriptDialogList, "wholeChatsFilePOS_N_ADJ", stopWordsDict, POS_list)

    POS_list = ['n','a','v']  # n - noun and a - adjective other possibilities: v -verb, r - adverb, 'other'
    writeWholeChatsToFile(transcriptDialogList, "wholeChatsFilePOS_N_ADJ_V", stopWordsDict, POS_list)
    
    return transcriptDialogList
    
t = main()  # start main running
