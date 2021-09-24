from pprint import pprint
from Parser import Parser
import util
import math
import time
import nltk

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex = []

    #of docs 
    n = 0

    #of docs with each term
    k = {}

    #Tidies terms
    parser = None


    def __init__(self, documents, representation_mode, relevance_mode):
        self.documentVectors=[]
        self.representation_mode = representation_mode
        self.relevance_mode = relevance_mode
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        #idf preprocessing
        if self.representation_mode == "TF-IDF":
            self.n = len(documents)
            #copy keys to new dict k with values 0
            self.k = dict.fromkeys(self.vectorKeywordIndex,0)
            for document in documents:
                document = self.parser.tokenise(document)
                document = self.parser.removeStopWords(document)
                #remove duplicate words
                document = util.removeDuplicates(document)
                for word in document:
                    self.k[word] += 1
            
        self.documentVectors = [self.makeVector(document) for document in documents]

        # print(self.vectorKeywordIndex)
        # print(self.documentVectors)


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)

        #Use TF Model
        if (self.representation_mode == "TF"):
            for word in wordList:
                if word in self.vectorKeywordIndex:
                    vector[self.vectorKeywordIndex[word]] += 1 

        #Use TF-IDF Model
        if self.representation_mode == "TF-IDF":
            #remove duplicate words
            wordList_ = list(dict.fromkeys(wordList)) 
            for word in wordList_:
                if word in self.vectorKeywordIndex:
                    tf = wordList.count(word)
                    idf = math.log10(self.n / (1+self.k[word]))
                    vector[self.vectorKeywordIndex[word]] = (tf*idf) 

        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings


    def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        if self.relevance_mode == "Cosine Similarity":
            ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        if self.relevance_mode == "Euclidean Distance":
            ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings
    

    def feedback_search(self, searchList, feedbackDoc):
        """ search for pseudo feedback """
        queryVector = self.buildQueryVector(searchList)
        #extract nouns&verbs by nltk
        tokens = nltk.word_tokenize(feedbackDoc)
        pos_tagged = nltk.pos_tag(tokens)
        wordList = [token for token, pos in pos_tagged if pos.startswith('N') or pos.startswith('V')]
        feedbackVector = self.makeVector(" ".join(wordList))
        #calculate new QueryVector (original query + 0.5*feedback vector)
        newQueryVector = [u+0.5*v for u, v in zip(queryVector, feedbackVector)]
        ratings = [util.cosine(newQueryVector, documentVector) for documentVector in self.documentVectors]
        return ratings