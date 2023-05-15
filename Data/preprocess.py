"""
    A file that defines the PreProcessor for the Corpus Interface,
    running this file on its own will allow for initial testing of the preprocessor
"""
import os, warnings
os.environ["CLTK_DATA"] = os.getcwd()+"/texts/"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=UserWarning)
    from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
    from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
    from cltk.lemmatize.lat import LatinBackoffLemmatizer
import re
from unidecode import unidecode

class PreProcessor():
    """
        A PreProcessing class so as to not instantiate multiple versions of the tokenizer in short
        succession needlessly in the Corpus Interface
    """

    def __init__(self):
        """ Instantiate only one tokenizer per preprocessor"""
        self.wt = WordTokenizer()
        self.st = SentenceTokenizer()
        self.lt = LatinBackoffLemmatizer()
    

    def preprocess(self, inputText: str, keepPunct: bool = True, shouldTokenize: bool = True, shouldLemma: bool = False) -> list[str]:
        """
            Preprocesses the given input txt. For now, shouldLemma should always be false, so that 
            the 'punc' error is not included.
        """
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

if __name__=="__main__":
    pp = PreProcessor()
    # testing the Greek, allows to see if the transliteration with the unicode works as intended
    text = 'atque haec Παρὰ τοῦ πάππου Οὐήρου τὸ καλόηθες καὶ ἀόργητον. abuterque puerve paterne nihil mecum. animiæger dicatur ut Seneca in Epistolis dixit.'
    text = "nautas viam puer. Vidit viam."
    print(pp.preprocess(text, True, True, False))