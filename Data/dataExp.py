"""
    A data exploration file
"""
import os, json
from texts.lat.text.lat_text_perseus.xml_to_json import cleanup_file_perseus_xml

# Directories for the data after fetch.py is run
PERSEUS_DATA_DIR = os.getcwd()+"/texts/lat/text/lat_text_perseus/"
LATIN_LIBRARY_DIR = os.getcwd()+"/texts/lat/text/lat_text_latin_library/"
ITALIAN_POETS_DIR = os.getcwd()+"/texts/lat/text/latin_text_poeti_ditalia/"
LATIN_TESSERAE_DIR = os.getcwd()+"/texts/lat/text/lat_text_tesserae/"
LATIN_GRAMMATICORUM_DIR = os.getcwd()+"/texts/lat/latin_text_corpus_grammaticorum_latinorum/"


class DataModel:
    def __init__(self):
        self.authors = []
        self.authorToWorks = []
        self.load_data()
        pass 
    
    def add_text(self, author, text):
        pass

    def load_perseus(self):
        for root, dirs, files in os.walk(PERSEUS_DATA_DIR):
            if root.split("/")[-1]!="opensource":continue
            print(root)
            for name in files:
                if name.split(".")[-1]!="json": continue
                #text = cleanup_file_perseus_xml(root+"/"+name)
                with open(root+"/"+name) as f:
                    data = json.load(f)
                
                print(data['TEI.2']['text'].keys())
                print(data['TEI.2']['text']['body'].keys())
                
                print("\t"+name)
                input()
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
        #self.load_perseus()
        self.load_latin_library()
        #self.load_italian_poets()
        #self.load_tesserae_corpus()
        #self.load_corpus_grammaticorum()
        
        pass
    

    def data_by_time_period(self):
        """ Return a distribution of the amount of text within given time periods """
        pass


if __name__=="__main__":
    dm = DataModel()