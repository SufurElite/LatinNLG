"""
    A data exploration file
"""
import os, json
from texts.lat.text.lat_text_perseus.xml_to_json import cleanup_file_perseus_xml, extract_xml_str, parse_chapter, parse_poems
from preprocess import PreProcessor
# Directories for the data after fetch.py is run
PERSEUS_DATA_DIR = os.getcwd()+"/texts/lat/text/lat_text_perseus/"
LATIN_LIBRARY_DIR = os.getcwd()+"/texts/lat/text/lat_text_latin_library/"
ITALIAN_POETS_DIR = os.getcwd()+"/texts/lat/text/latin_text_poeti_ditalia/"
LATIN_TESSERAE_DIR = os.getcwd()+"/texts/lat/text/lat_text_tesserae/texts/"
LATIN_GRAMMATICORUM_DIR = os.getcwd()+"/texts/lat/latin_text_corpus_grammaticorum_latinorum/"


class DataEngModel:
    def __init__(self):
        # constants
        self.FILE_TYPE_PARSERS = [parse_chapter, parse_poems]
        self.includeLineBreaks = False
        self.PrePro = PreProcessor()
        self.authorToWorks = {}
        self.load_data()
    
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
        This function finds the the Longest Common Subphrase between the two preprocessed texts, using dynamic programming,
        that then can be used to determine if two texts are sufficiently identical to not include them again 
        """
    
        DP = [[0 for k in range(len(textTwo)+1)] for l in range(len(textOne)+1)]
 
        # To store the length of
        # longest common subphrase
        lcsLength = 0
 
        for i in range(len(textOne) + 1):
            for j in range(len(textTwo) + 1):
                if (i == 0 or j == 0):
                    DP[i][j] = 0
                elif (textOne[i-1] == textTwo[j-1]):
                    DP[i][j] = DP[i-1][j-1] + 1
                    lcsLength = max(lcsLength, DP[i][j])
                else:
                    DP[i][j] = 0
        return (lcsLength/len(textOne))>simPercent

    def load_perseus(self):
        """ A modification of the xml_to_json main function"""
        for root, dirs, files in os.walk(PERSEUS_DATA_DIR):
            dirValues = root.split("/")
            if dirValues[-1]!="opensource":continue
            print(dirValues[-2].lower())
            texts = []
            for name in files:
                if name.split(".")[-1]!="xml" or name.find("_lat")==-1: continue
                fpath = root+"/"+name
                text = ""

                xml_str = cleanup_file_perseus_xml(fpath)
                print(fpath)
                tei = extract_xml_str(xml_str)
                
                for file_parser in self.FILE_TYPE_PARSERS:
                    try:
                        dict_object = file_parser(tei, fpath)
                        print(dict_object)
                        if str(dict_object) == "<class 'AssertionError'>":
                            continue
                        #if dict_object['text'] == {}:
                        if dict_object.get('text') == {}:
                            continue

                        print("Parser '{}' worked! Moving to next file …".format(file_parser))
                        text = dict_object.get('text')
                        break

                    except AttributeError as attrib_err:
                        pass
                print(name)
                print(text)
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
        for root, dirs, files in os.walk(LATIN_TESSERAE_DIR):
            dirValues = root.split("/")
            if dirValues[-1]=="metadata": continue
            for name in files:
                nameParts = name.split(".")
                if nameParts[-1]!="tess": continue
                print(name)
                author = nameParts[0].lower().replace("_"," ")
                text = ""

                with open(root+"/"+name) as f:
                    if not self.includeLineBreaks:
                        data = f.read().split("\n")
                        del data[len(data)-1]
                    else:
                        data = f.readlines()

                for lineNum in range(len(data)):    
                    line = data[lineNum]
                    text+= line[line.find(">")+2:]
                    if lineNum!=len(data)-1:
                        text+=" "
                self.add_text(author, text)  
    
    def load_corpus_grammaticorum(self):
        pass 

    def load_data(self):
        self.load_tesserae_corpus()
        #self.load_perseus()
        #self.load_latin_library()
        #self.load_italian_poets()
        #self.load_corpus_grammaticorum()

        # now print information about the corpus
        self.corpus_overview()
        
    def corpus_overview(self):
        """ Just a helper function to show some information about the corpus """
        for author in self.authorToWorks.keys():
            lengthOfTexts = 0
            for text in self.authorToWorks[author]:
                lengthOfTexts+=len(text)
            print("Author {} had {} pieces of work with a total of {} characters of text".format(author, len(self.authorToWorks[author]),lengthOfTexts))


    def data_by_time_period(self):
        """ Return a distribution of the amount of text within given time periods """
        pass


if __name__=="__main__":
    dm = DataEngModel()
    """
    text1 = "multa quoque et bello passus, dum conderet urbem, inferretque deos Latio, genus unde Latinum, Albanique patres, atque altae moenia Romae. Musa, mihi causas memora, quo numine laeso, quidve dolens, regina deum tot volvere casus"
    text2 = "litora, multum ille et terris iactatus et alto vi superum saevae memorem Iunonis ob iram; multa quoque et bello passus, dum conderet urbem, inferretque deos Latio, genus unde Latinum, Albanique patres, atque altae moenia Romae. Musa, mihi causas memora, quo numine laeso, quidve dolens, regina deum tot volvere casus"
    text1 = dm.PrePro.preprocess(text1)
    text2 = dm.PrePro.preprocess(text2)
    print(dm.similarity_identification(text1,text2))
    """
