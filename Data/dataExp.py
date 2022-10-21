"""
    A data exploration file
"""
import os, json
from texts.lat.text.lat_text_perseus.xml_to_json import cleanup_file_perseus_xml
from preprocess import PreProcessor
# Directories for the data after fetch.py is run
PERSEUS_DATA_DIR = os.getcwd()+"/texts/lat/text/lat_text_perseus/"
LATIN_LIBRARY_DIR = os.getcwd()+"/texts/lat/text/lat_text_latin_library/"
ITALIAN_POETS_DIR = os.getcwd()+"/texts/lat/text/latin_text_poeti_ditalia/"
LATIN_TESSERAE_DIR = os.getcwd()+"/texts/lat/text/lat_text_tesserae/"
LATIN_GRAMMATICORUM_DIR = os.getcwd()+"/texts/lat/latin_text_corpus_grammaticorum_latinorum/"


class DataEngModel:
    def __init__(self):
        self.PrePro = PreProcessing()
        self.authorToWorks = {}
        self.load_data()
        pass 
    
    def add_text(self, author: str, text: str) -> bool:
        """ Adds a particular text by a given author to the corpus, if the text does not already exist
        returns a boolean related to the success of adding the text"""
        if author not in self.authorToWorks:
            self.authorToWorks[author] = []
        ppText = self.PrePro.preprocess(text, keepPunct = False)
        for og_texts in self.authorToWorks[author]:
            # og_texts[0] = the version without punctuation
            if self.similarity_identification(og_texts[0], ppText):
                print("Found identical texts for {}, \n\t{}".format(author,text))
                return False
        punctText = self.PrePro.preprocess(text, keepPunct = True)
        self.authorToWorks[author].append([ppText,punctText])
        return True

    def similarity_identification(self, textOne, textTwo, simPercent = .7):
        """
        This function takes in three arguments:
             - two preprocessed texts as lists to identify if they're the same text
               (textOne is the original already stored, textTwo is the new text to be compared)
             - a similarity percentage - the percentage of text that must overlap to be identified as the same
        This function finds the the Longest Common Substring between the two preprocessed texts, using dynamic programming,
        that then can be used to determine if two texts are sufficiently identical to not include them again 
        """
    
        DP = [[0 for k in range(len(textTwo)+1)] for l in range(len(textOne)+1)]
 
        # To store the length of
        # longest common substring
        lcsLength = 0
 
        # Following steps to build
        # LCSuff[m+1][n+1] in bottom up fashion
        for i in range(len(textOne) + 1):
            for j in range(len(textTwo) + 1):
                if (i == 0 or j == 0):
                    DP[i][j] = 0
                elif (textOne[i-1] == textTwo[j-1]):
                    DP[i][j] = DP[i-1][j-1] + 1
                    lcsLength = max(lcsLength, DP[i][j])
                else:
                    DP[i][j] = 0
        
        return lcsLength/len(textOne)
    
    def load_perseus(self):
        for root, dirs, files in os.walk(PERSEUS_DATA_DIR):
            dirValues = root.split("/")
            if dirValues[-1]!="opensource":continue
            print(dirValues[-2].lower())
            texts = []
            for name in files:
                if name.split(".")[-1]!="xml" or name.find("_lat")==-1: continue
                #text = cleanup_file_perseus_xml(root+"/"+name)
                #with open(root+"/"+name) as f:
                #    data = json.load(f)
                print(cleanup_file_perseus_xml(root+"/"+name))
                input()
                texts.append(name)
                #print(data['TEI.2']['text'].keys())
                #print(data['TEI.2']['text']['body'].keys())
                 
                #print("\t"+name)
            print(texts)
            input()
        pass 

    def load_latin_library(self):
        for root, dirs, files in os.walk(LATIN_LIBRARY_DIR):
            print(root)
            for name in files:
                if name.split(".")[-1]!="txt": continue
                print("\t"+name)
            input() 
        pass

    def load_italian_poets(self):
        pass

    def load_tesserae_corpus(self):
        pass 
    
    def load_corpus_grammaticorum(self):
        pass 

    def load_data(self):
        self.load_perseus()
        #self.load_latin_library()
        #self.load_italian_poets()
        #self.load_tesserae_corpus()
        #self.load_corpus_grammaticorum()
        
    

    def data_by_time_period(self):
        """ Return a distribution of the amount of text within given time periods """
        pass


if __name__=="__main__":
    dm = DataEngModel()
    text1 = "Rufus was here".lower().split(" ")
    text2 = "I guess here is Rufus".lower().split(" ")
    print(dm.similarity_identification(text1,text2))
