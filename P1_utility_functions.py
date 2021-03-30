""" File:  P1_utility_functions.py
    Utility functions used in Phase 1 Pre-processing of the raw chat (.csv) file
"""
import os.path
import re
import nltk
import gensim
from nltk.stem import WordNetLemmatizer


def getStopWords(stopWordFileName):
    """Reads stop-words text file which is assumed to have one word per line.
       Returns stopWordDict.
    """
    stopWordDict = {}
    stopWordFile = open(stopWordFileName, 'r')

    for line in stopWordFile:
        word = line.strip().lower()
        stopWordDict[word] = None
        
    return stopWordDict

def getFileName(prompt):
    """Prompts the user for a valid file which it returns.
    """
    while True:
        fileName = input(prompt+" ")
        if os.path.exists(fileName):
            return fileName
        else:
            print("File not found! Make sure that the file is inside this directory.")

def readRawChats(inFile):
    """
        Reads .csv file and split into transcripts by splitting on the Timestamp which includes the Date.
        The returned transcriptList is a list-of-lists where each "outer" list item contains information about
        a single chat.  
    """

    inFile = open(inFile, "r")  # NOTE .csv file assumed to have column-headings line

    dateAtStartCount = 0
    transcriptList = []
    currentTranscriptLines = []

    for line in inFile:
        frontOfLine = line[:6]
        if frontOfLine.count("/") == 2:
            dateAtStartCount += 1
            if dateAtStartCount == 1: #ignore header line
                currentTranscriptLines = [line.strip()]
            else:
                transcriptList.append(currentTranscriptLines)
                currentTranscriptLines = [line.strip()]
        else:
            currentTranscriptLines.append(line.strip())
    transcriptList.append(currentTranscriptLines)
    
    return transcriptList


def findInitialQuestion(transList, transIndex):
    """
        Takes in transList which is a list of strings containing the information about a single chat.
        The index 0 string will contain the Initial Question field, which it returns if it exists; otherwise
        None is returned."
    """
    
    firstCommaIndex = transList[0].find(",")
    if firstCommaIndex == -1:
        print("First comma not found")
        return None
    else:
        secondCommaIndex = transList[0].find(",",firstCommaIndex+1)
        if secondCommaIndex == -1:
            print("Second comma not found")
            return None
        else:
            thirdCommaIndex = transList[0].find(",",secondCommaIndex+1)
            if thirdCommaIndex == -1:
                thirdCommaIndex = len(transList[0])-1
           
            #print(secondCommaIndex, thirdCommaIndex)
            if secondCommaIndex + 1 == thirdCommaIndex:
                return None
            else:
                return transList[0][secondCommaIndex+1:thirdCommaIndex]

            
def generateTranscriptDialogList(trans):
    
    transcriptDialogList = []
    transStr = " ".join(trans)  # merge transcript back to a single string

    #split by time-stamps to get a dialogList
    transTimeIndexList = []
    for index in range(2,len(transStr)-6):
        if transStr[index] == ":" and transStr[index+3] == ":" and transStr[index+1:index+3].isdigit() and transStr[index+4:index+6].isdigit():
            transTimeIndexList.append(index-2)
    dialogList = []
    for i in range(len(transTimeIndexList)-1):
        dialogList.append(transStr[transTimeIndexList[i]:transTimeIndexList[i+1]])
    if len(transTimeIndexList) == 0:
        dialogList.append(transStr)
    else:
        dialogList.append(transStr[transTimeIndexList[-1]:])
    
    return dialogList    

def findInitialQuestionInDialog(dialogList, chatIndex):
    """ If the 'Initial question' column in the .csv file was empty, this function is called
        to find and return the initial question from the chat dialog."""

    for i in range(len(dialogList)):
        helpYouCount = dialogList[i].lower().count("help you")
        welcomeCount = dialogList[i].lower().count("welcome")
        infoDeskCount = dialogList[i].lower().count("info desk")
        try:
            if helpYouCount == 0 and welcomeCount == 0 and infoDeskCount == 0 and len(dialogList[i]) >= 40:
                return dialogList[i]
                
        except:
            print("\n\nNO QUESTION FOUND! ",chatIndex)
            break

def removeTags(fileStr):
    """
        Removes all tags from the chat that start with '<xyz' and end with '</xyz'.
    """
    current = 0
    while True:
        #print("Next char:",fileStr[current])
        openAngleBracketIndex = fileStr.find('<',current)
        if openAngleBracketIndex == -1:
            break
        spaceIndex = fileStr.find(' ', openAngleBracketIndex+1)
        if spaceIndex == -1:
            break
        else:
            current = spaceIndex
        endStr = "</"+fileStr[openAngleBracketIndex+1:spaceIndex]+'>'

        endIndex = fileStr.find(endStr, spaceIndex)
        if endIndex == -1:
            current = spaceIndex
        else:
            endIndex = endIndex+len(endStr)

            #print(openAngleBracketIndex, endStr, endIndex+len(endStr))
            fileStr = fileStr[:openAngleBracketIndex]+ \
                      fileStr[endIndex:]
            #print(fileStr)
            current = openAngleBracketIndex
    return fileStr


"""
NOTE: The nltk.pos_tag function returns the Penn Treebank tag for the word but we just want
whether the word is a noun, verb, adjective or adverb. We need a short simplification routine to translate from
the Penn tag to a simpler tag.
"""
def simplify(penn_tag):
    """ Simplify Penn tags to n (NOUN), v (VERB), a (ADJECTIVE) or r (ADVERB)"""
    pre = penn_tag[0]
    
    if pre == 'J':
        return 'a'
    elif pre == 'R':
        return 'r'
    elif pre == 'V':
        return 'v'
    elif pre == 'N':
        return 'n'
    else:
        return 'r'
        return 'other'

def preprocess(text, stop_words, POS_list):
    """ Preprocesses the text to remove stopwords, lemmatizes each word and only includes
        words that are POS in the global POS_LIST"""

    toks = gensim.utils.simple_preprocess(str(text), deacc=True)
    wn = WordNetLemmatizer()
    return [wn.lemmatize(tok, simplify(pos)) for tok, pos in nltk.pos_tag(toks)
            if tok not in stop_words and simplify(pos) in POS_list]
        
def writeInitialQuestion(chatIndexInCSV, questionFile,  wholeChatsFileTxt, question, questionCount, stopWordsDict, POS_list):
    """ Write a cleaned up version of the initial question to the question file. """
    lemmatizer = nltk.WordNetLemmatizer()
    cleanQuestion = ""
    question = question.lower()

    colonCount = question.count(":")

    if colonCount >= 3:  # time-stamp ##:##:## - person: question
        colonOneIndex = question.find(":")
        colonTwoIndex = question.find(":", colonOneIndex+1)
        colonThreeIndex = question.find(":", colonTwoIndex+1)
        question = question[colonThreeIndex+1:]
    elif colonCount >= 1:
        colonOneIndex = question.find(":")
        question = question[colonOneIndex+1:]
        
    question = question.replace('&#x27;', "'")
    question = question.replace('&#x2F;', " ")
    question = question.replace('&nbsp;', " ")
    question = question.replace('&quot;','"')

    ### HERE CLEAN UP <xyz ......</xyz>, e.g., <a href.....</a>, <span ... </span>

    question = removeTags(question)
    question = question.replace('.','Z')
    question = question.replace('!','Z')
    question = question.replace('?','Z')
    
    masterWordList = []
    sentenceList = question.split("Z")
    for question in sentenceList:
        wordList = question.split()
        cleanQuestion = ""
        for word in wordList:
            cleanWord = ""
            for char in word:
                if char >= 'a' and char <= 'z':
                    cleanWord += char
            if len(cleanWord) > 0 and len(cleanWord) < 30:  #upper bound to eliminate url's
                cleanQuestion += lemmatizer.lemmatize(cleanWord) + " "
        pos_wordList = preprocess(cleanQuestion, stopWordsDict, POS_list)
          
        masterWordList.extend(pos_wordList)

    chatCleaned = " ".join(masterWordList)
    if len(chatCleaned) > 0:
        questionFile.write(chatCleaned)
        wholeChatsFileTxt.write(chatCleaned)
        questionCount += 1
    return questionCount

def writeChatDialog(excelLineNumber, wholeChatsFile,  wholeChatsFileTxt, dialogList, stopWordsDict, POS_list):
    """ Writes a chat's dialog to a line in the text file. """
    for i in range(len(dialogList)):
      
        writeInitialQuestion(excelLineNumber, wholeChatsFile,  wholeChatsFileTxt, dialogList[i], 0, stopWordsDict, POS_list)
        wholeChatsFile.write(" ")  # separate end of this line with start of next line
        wholeChatsFileTxt.write(" ")  # separate end of this line with start of next line
        
   
def writeWholeChatsToFile(transcriptDialogList, dataFileName, stopWordsDict, POS_list):
    """ Writes a whole chat's dialog one per line to a text file.  Removed from
        the line of text is:
        1) time-stamps and names:  e.g., '13:45:42 - Jordan:'
        2) all punctuations
    """

    wholeChatsFile = open(dataFileName+".csv", "w")
    wholeChatsFileTxt = open(dataFileName+".txt", "w")
    wholeChatsCount = 0
    for transcriptDialog in transcriptDialogList:

        if transcriptDialog[1] is not None:
            wholeChatsFile.write(str(transcriptDialog[0])+",")

            # check to see if initial question is already in the chat dialog
            timeStampAndNameList = re.findall(r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9] - [\w\s]+:', transcriptDialog[1])
            
            if len(timeStampAndNameList) == 0:  # no time-stamp so from 'initial question' column of .csv
                # write initial question to file since it is not part of the chat dialog
                writeInitialQuestion(transcriptDialog[0], wholeChatsFile, wholeChatsFileTxt, transcriptDialog[1], 0, stopWordsDict, POS_list)
                wholeChatsFile.write(" ")
                wholeChatsFileTxt.write(" ")
            writeChatDialog(transcriptDialog[0],wholeChatsFile,  wholeChatsFileTxt, transcriptDialog[2], stopWordsDict, POS_list)
            
            #wholeChatsFile.write("\n")
            wholeChatsCount += 1
            wholeChatsFile.write("\n")
            wholeChatsFileTxt.write("\n")
    print("Whole Chats Count:", wholeChatsCount, "written to",dataFileName+".txt")
    wholeChatsFile.close()
    wholeChatsFileTxt.close()

def writeQuestionsOnlyToFile(transcriptDialogList, dataFileName, stopWordsDict, POS_list):
    """ Writes only the initial questions one per line to a text file. 
    """
    questionFile = open(dataFileName+".csv", "w")
    questionTxtFile = open(dataFileName+".txt", "w")
    questionCount = 0
    for transcriptDialog in transcriptDialogList:
        if transcriptDialog[1] is not None:
            currentCount = questionCount
            questionCount = writeInitialQuestion(transcriptDialog[0], questionFile, questionTxtFile, transcriptDialog[1], questionCount, stopWordsDict, POS_list)
            if currentCount < questionCount:
                questionFile.write("\n")
                questionTxtFile.write("\n")
    print("Total Question Count:", questionCount, "written to",dataFileName+".txt")
    questionFile.close()
    questionTxtFile.close()

