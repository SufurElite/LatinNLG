"""
    A data exploration file
"""
import os, json, re, pickle
from .preprocess import PreProcessor
from .fetch import text_retrieval
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors

# Directories for the data after fetch.py is run
LATIN_BASE_DIR = os.getcwd()+"/Data/texts/lat/text/"
LATIN_LIBRARY_DIR = LATIN_BASE_DIR+"lat_text_latin_library/"
ITALIAN_POETS_DIR = LATIN_BASE_DIR+"latin_text_poeti_ditalia/cltk_json/"
LATIN_TESSERAE_DIR = LATIN_BASE_DIR+"lat_text_tesserae/texts/"

class CorpusInterface:
    def __init__(self, corpus_name="corpus.pickle", shouldTokenize:bool = True, particular_data: list = ["tesserae", "italian_poets", "latin_library"]):
        self.OUR_CORPUS_LOC=os.getcwd()+"/Data/"+corpus_name
        # for the LatinBERT text encoding, tokenization will happen within the model
        self.shouldTokenize = shouldTokenize
        self.includeLineBreaks = False
        self.PrePro = PreProcessor()
        self.authorToWorks = {}
        self.authorToColours = {}
        self.load_data(particular_data)
        
        
        
    def add_text(self, author: str, text: str) -> bool:
        """ Adds a particular text by a given author to the corpus, if the text does not already exist
        returns a boolean related to the success of adding the text"""
        if author not in self.authorToWorks:
            self.authorToWorks[author] = []
        
        text = self.PrePro.preprocess(text, keepPunct = False, shouldTokenize = self.shouldTokenize)  
        self.authorToWorks[author].append([text,None])
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

    def load_latin_library(self):
        for root, dirs, files in os.walk(LATIN_LIBRARY_DIR):
            print(root)
            author = root.split("/")[-1].strip()
            if author == "git": continue

            print("author : {}".format(author))
            for name in files:
                fileName = name.split(".")
                if fileName[-1]!="txt": continue
                if root==LATIN_LIBRARY_DIR:
                    numIdx = re.search(r"\d", fileName[0])
                    if numIdx:
                        author = fileName[0][:numIdx.start()].strip()
                    else:
                        author = fileName[0].strip()
                    print("author : {}".format(author))
                print("\t\t"+name)
                # RIGHT NOW JUST A SKIP TO NOT WORRY ABOUT DUPLICATES UNTIL I USE THE OTHER METHODOLOGY
                if author in self.authorToWorks: continue
                if author =="": continue
                
                text = []
                with open(root+"/"+name) as f:
                    if not self.includeLineBreaks:
                        data = f.read().split("\n")
                        del data[len(data)-1]
                    else:
                        data = f.readlines()
                
                lastEmpty = 0
                for lineNum in range(len(data)):
                    line = data[lineNum].strip()
                    if not line:
                        lastEmpty = lineNum
                        continue
                        
                    text.append(line)
                    if lineNum==len(data)-1:
                        # find how many lines to delete from the text
                        diffInLength = len(data)-lastEmpty-1
                        del text[-diffInLength:]
                        # will add algorithmically more words to search for if there are any English sentences at the 
                        # end of the text
                        if text!=[] and text[len(text)-1].lower().find("prepared")>-1:
                            del text[len(text)-1]
                if text==[]: continue
                text = ' '.join(text)
                self.add_text(author, text)
        pass
    
    def load_italian_poets(self):
        for root, dirs, files in os.walk(ITALIAN_POETS_DIR):
            for name in files:
                if name.split(".")[-1]!="json": continue
                print(root+"/"+name)
                with open(root+"/"+name) as f:
                    data = json.load(f)
                
                author = data['author'].lower().replace(" ",'')
                # RIGHT NOW JUST A SKIP TO NOT WORRY ABOUT DUPLICATES UNTIL I USE THE OTHER METHODOLOGY
                if author in self.authorToWorks: continue
                
                text = []
                for i in data['text'].keys():
                    text.append(data['text'][i])
                text = " ".join(text)
                
                self.add_text(author, text)  


    def load_tesserae_corpus(self):
        for root, dirs, files in os.walk(LATIN_TESSERAE_DIR):
            dirValues = root.split("/")
            if dirValues[-1]=="metadata": continue
            for name in files:
                nameParts = name.split(".")
                if nameParts[-1]!="tess": continue
                print(name)
                author = nameParts[0].lower().replace("_","")
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
        for root, dirs, files in os.walk(LATIN_GRAMMATICORUM_DIR):
            for name in files:
                if name.split(".")[-1]!="json": continue
                print(root+"/"+name)
                with open(root+"/"+name) as f:
                    data = json.load(f)
                text = data['text']
                print(text)
                print(type(text))
                input()
                author = data['author'].lower().replace(" ",'')
                self.add_text(author, text)  

    def load_new_data(self, particular_data: list = ["tesserae", "italian_poets", "latin_library"]):
        if "tesserae" in particular_data:
            print("Loading Tesserae")
            self.load_tesserae_corpus()
        if "italian_poets" in particular_data:
            print("Loading Italian Poets")
            self.load_italian_poets()
        if "latin_library" in particular_data:
            print("Loading Latin Library")
            self.load_latin_library()
        
    def save_corpus(self):
        with open(self.OUR_CORPUS_LOC, "wb") as f:
            pickle.dump(self.authorToWorks, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_existing_corpus(self):
        with open(self.OUR_CORPUS_LOC, 'rb') as f:
            self.authorToWorks = pickle.load(f)

    def load_data(self, particular_data: list = ["tesserae", "italian_poets", "latin_library"]):
        # first check if we have loaded the data using the fetch function
        if not os.path.exists(LATIN_BASE_DIR):
            print("Did not find the downloaded corpus. Did you run fetch.py? Now calling text_retrieval from fetch.py")
            text_retrieval()
        # check if we have already created the dataset previously
        if os.path.exists(self.OUR_CORPUS_LOC):
            print("Found the existing corpus")
            self.load_existing_corpus()
        else: 
            self.load_new_data(particular_data)
            # save new data
            self.save_corpus()

        # now print information about the corpus
        self.corpus_overview()
        self.associate_author_to_colour()
    
    def associate_author_to_colour(self):
        # create a colour map
        
        authors = list(self.authorToWorks.keys())
        authors.sort()
        cm = plt.get_cmap('gist_rainbow')
        cNorm  = colors.Normalize(vmin=0, vmax=len(authors)-1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        for i in range(len(authors)):
            self.authorToColours[authors[i]] = scalarMap.to_rgba(i)

    def corpus_overview(self, saveText: bool = False):
        """ Just a helper function to show some information about the corpus """
        authorStats = ""
        authors = ""
        sorted_auths = sorted(self.authorToWorks.keys())
        for author in sorted_auths:
            authors+=author+"\n"
            lengthOfTexts = 0
            for text in self.authorToWorks[author]:
                lengthOfTexts+=len(text[0])
            cur_auth_stat = "{} had {} pieces of work with a total of {} characters of text".format(author, len(self.authorToWorks[author]),lengthOfTexts)
            print(cur_auth_stat)
            authorStats+=cur_auth_stat+"\n"
        if saveText:
            with open("author_stats.txt", "w+") as f:
                f.write(authorStats)
            with open("authors.txt", "w+") as f:
                f.write(authors)

    def get_authors(self):
        return self.authorToWorks.keys()

    def get_author_color(self, author):
        return self.authorToColours[author]

    def get_authors_by_text_size(self, characterCount: bool = True):
        def sort_tuple(tup):
            #https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
            return sorted(tup, key = lambda x: x[1], reverse=True)
 
        values = []
        for author in self.authorToWorks:
            num = 0
            if characterCount:
                for j in range(len(self.authorToWorks[author])):
                    num+=len(self.authorToWorks[author][j][0])
            else:
                num+=len(self.authorToWorks[author])
            values.append((author,num))
        values = sort_tuple(values)
        return values

    def get_text_for_author(self, author):
        text = []
        assert(author in self.authorToWorks)
        for txt in self.authorToWorks[author]:
            text+=txt[0]
        return " ".join(text)

    def get_data(self, n_authors: int = 50, keepPunct: bool = False, max_words: int = -1):
        """ return the corpus's data that can be used by a model 
            , max_docs: int = 4, max_words: int = 500
        """
        
        texts = []
        authors = []
        
        values = self.get_authors_by_text_size()

        for i in range(n_authors):
            author = values[i][0]
            for j in range(len(self.authorToWorks[author])):
                text = self.authorToWorks[author][j][0]
                if max_words!=-1:
                    text = text.split(" ")
                    if len(text)>max_words:
                        text = text[:200]
                    text = " ".join(text)
                texts.append(text)
                authors.append(author)
        return texts, authors

    def lexical_diversity(self, authors):
        """
            This will measure the lexical diversity of a subset of the authors provided.
        """
        values = []
        for author in authors:
            totalText = []
            initialLength = 0
            for text in self.authorToWorks[author]:
                totalText+=text[0]
                initialLength+=len(text[0])
            lexicalDiversity = len(totalText)/len(set(totalText))
            values.append((author, lexicalDiversity))
        return sorted(values, key = lambda x: x[1], reverse=True)

    def data_by_time_period(self):
        """ Return a distribution of the amount of text within given time periods """
        pass