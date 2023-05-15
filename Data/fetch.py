"""
    A file to retrieve all the necessary data files
"""

# check if the directory exists yet
import os
if not os.path.exists(os.getcwd()+"/Data"):
    os.mkdir(os.getcwd()+"/Data") 
if not os.path.exists(os.getcwd()+"/Data/texts"):
    os.mkdir(os.getcwd()+"/Data/texts") 
os.environ["CLTK_DATA"] = os.getcwd()+"/Data/texts/"

from cltk import NLP
from cltk.data.fetch import FetchCorpus
from cltk.embeddings.embeddings import Word2VecEmbeddings as W2VE


def text_retrieval() -> None:
    corpaDownloader = FetchCorpus(language="lat")
    print("Retrieving Perseus Texts")
    corpaDownloader.import_corpus("lat_text_perseus")
    print("Retrieving Latin Library Texts")
    corpaDownloader.import_corpus("lat_text_latin_library")
    print("Retrieving \"Italian Poets in Latin\" Texts")
    corpaDownloader.import_corpus("latin_text_poeti_ditalia")
    print("Retrieving CLTK Tesserae Latin Corpus")
    corpaDownloader.import_corpus("lat_text_tesserae")
    print("Retrieving the Grammaticorum Latinorum Texts")
    corpaDownloader.import_corpus("latin_text_corpus_grammaticorum_latinorum")
    print("Installing Language Models")
    corpaDownloader.import_corpus("lat_models_cltk")
    print("Downloading Latin word2vec model")
    w2v = W2VE(iso_code="lat")
    