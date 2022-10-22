import os
os.environ["CLTK_DATA"] = os.getcwd()+"/texts/"

from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
import re

class PreProcessor():
    """ A PreProcessing class so as to not instantiate multiple versions of the tokenizer in short
    succession needlessly"""
    def __init__(self):
        self.wt = WordTokenizer()
        self.st = SentenceTokenizer()
    

    def preprocess(self, inputText: str, keepPunct: bool = True) -> [str]:
        text = []
        inputText = inputText.lower()
        text = self.wt.tokenize("atque haec abuterque puerve paterne nihil", enclitics=["que"])
        sents = self.st.tokenize(inputText)

        for sent in sents:
            tmpSent = []
            if not keepPunct:
                sent = re.sub(r'[^\w\s]', '', sent)
            wordToks = self.wt.tokenize(sent)
            text+=wordToks
            
        return text 

if __name__=="__main__":
    pp = PreProcessor()
    text = 'atque haec abuterque puerve paterne nihil mecum'
    print(pp.preprocess(text))
