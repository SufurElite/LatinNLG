import os
from cltk import NLP
from cltk.data.fetch import FetchCorpus

def textRetrieval():
    os.environ["CLTK_DATA"] = "/Data/texts/"
    corpaDownloader = FetchCorpus(language="lat")
    # Perseus Retrieval
    print("Retrieving Perseus Texts")
    corpaDownloader.import_corpus("lat_text_perseus")


if __name__=="__main__":
    textRetrieval()
