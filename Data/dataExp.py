"""
    A data exploration file
"""
import os, json, re, pickle
from texts.lat.text.lat_text_perseus.xml_to_json import cleanup_file_perseus_xml, extract_xml_str, parse_chapter, parse_poems
from preprocess import PreProcessor
from fetch import text_retrieval
# Directories for the data after fetch.py is run
LATIN_BASE_DIR = os.getcwd()+"/texts/lat/text/"
PERSEUS_DATA_DIR = LATIN_BASE_DIR+"lat_text_perseus/"
LATIN_LIBRARY_DIR = LATIN_BASE_DIR+"lat_text_latin_library/"
ITALIAN_POETS_DIR = LATIN_BASE_DIR+"latin_text_poeti_ditalia/cltk_json/"
LATIN_TESSERAE_DIR = LATIN_BASE_DIR+"lat_text_tesserae/texts/"
LATIN_GRAMMATICORUM_DIR = LATIN_BASE_DIR+"latin_text_corpus_grammaticorum_latinorum/"

OUR_CORPUS_LOC = os.getcwd()+"/corpus.pickle"

class CorpusInterface:
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
        self.authorToWorks[author].append([ppText,text])
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

    def load_new_data(self):
        self.load_tesserae_corpus()
        self.load_italian_poets()
        self.load_latin_library()
        
        #self.load_perseus()
        #self.load_corpus_grammaticorum()

        # Once we've loaded in all the data available, determine if there are any identical texts for the authors
        # this is quite a slow methodology
        
        authors = sorted(self.authorToWorks.keys())
        for author in authors:
            print("Currently on {}".format(author))
            if len(self.authorToWorks[author])==1: continue
            removeList = []
            for i in range(len(self.authorToWorks[author])):
                if i in removeList: continue
                for j in range(i+1, len(self.authorToWorks[author])):
                    print("{} | {}:{}".format(author,str(i),str(j)))
                    if j in removeList: continue
                    if self.similarity_identification(self.authorToWorks[author][i][0], self.authorToWorks[author][j][0]):
                        print("Found identical texts for {}".format(author))
                        if len(self.authorToWorks[author][i][0])<len(self.authorToWorks[author][j][0]):
                            removeList.append(i)
                        else:
                            removeList.append(j)
                if i not in removeList:
                    punctText = self.PrePro.preprocess(self.authorToWorks[author][i][1], keepPunct = True)
                    self.authorToWorks[author][i][1] = punctText
            for i in removeList:
                del self.authorToWorks[author][i]   

    def save_corpus(self):
        with open(OUR_CORPUS_LOC, "wb") as f:
            pickle.dump(self.authorToWorks, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_existing_corpus(self):
        with open(OUR_CORPUS_LOC, 'rb') as f:
            self.authorToWorks = pickle.load(f)

    def load_data(self):
        # first check if we have loaded the data using the fetch function
        if not os.path.exists(LATIN_BASE_DIR):
            print("Did not find the downlaoded corpus. Did you run fetch.py? Now calling text_retrieval from fetch.py")
            text_retrieval()
        # check if we have already created the dataset previously
        if os.path.exists(OUR_CORPUS_LOC):
            print("Found the existing corpus")
            self.load_existing_corpus()
        else: 
            self.load_new_data()
            # save new data
            self.save_corpus()

        # now print information about the corpus
        self.corpus_overview()
        
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

    def get_authors_by_text_size(self, characterCount: bool = True):
        def sort_tuple(tup):
            #https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
            # reverse = None (Sorts in Ascending order)
            # key is set to sort using second element of
            # sublist lambda has been used
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


if __name__=="__main__":
    ci = CorpusInterface()