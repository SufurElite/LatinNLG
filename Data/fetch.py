import os
os.environ["CLTK_DATA"] = os.getcwd()+"/texts/"

from cltk import NLP
from cltk.data.fetch import FetchCorpus
from texts.lat.text.lat_text_perseus.xml_to_json import cleanup_file_perseus_xml


def textRetrieval():
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
if __name__=="__main__":
    print(cleanup_file_perseus_xml(os.getcwd()+"/texts/lat/text/lat_text_perseus/Horace/opensource/hor.ap_lat.xml"))
