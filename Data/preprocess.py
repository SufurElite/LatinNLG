import os, warnings
os.environ["CLTK_DATA"] = os.getcwd()+"/Data/texts/"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=UserWarning)
    from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
    from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
    from cltk.lemmatize.lat import LatinBackoffLemmatizer
import re
from unidecode import unidecode

class PreProcessor():
    """ A PreProcessing class so as to not instantiate multiple versions of the tokenizer in short
    succession needlessly"""
    def __init__(self):
        self.wt = WordTokenizer()
        self.st = SentenceTokenizer()
        self.lt = LatinBackoffLemmatizer()
    

    def preprocess(self, inputText: str, keepPunct: bool = True, shouldTokenize: bool = True, shouldLemma: bool = True) -> list[str]:
        text = []
        # include only unicode characters
        inputText = unidecode(inputText).lower()
        if shouldTokenize:
            sents = self.st.tokenize(inputText)
            for sent in sents:
                tmpSent = []
                if not keepPunct:
                    sent = re.sub(r'[^\w\s]', '', sent)
                wordToks = self.wt.tokenize(sent)
                if shouldLemma:
                    res = self.lt.lemmatize(wordToks)
                    lemmToks = [''.join([i for i in tmp[1] if not i.isdigit()]) for tmp in res]
                    text+=lemmToks
                else:
                    text+=wordToks
            return text
        return inputText 

if __name__=="__main__":
    pp = PreProcessor()
    text = 'atque haec Παρὰ τοῦ πάππου Οὐήρου τὸ καλόηθες καὶ ἀόργητον. abuterque puerve paterne nihil mecum. animiæger dicatur ut Seneca in Epistolis dixit.'
    text = "nautas viam puer"
    print(pp.preprocess(text, False))
