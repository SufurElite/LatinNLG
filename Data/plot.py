"""
    File to plot the findings from the Data Exploration file
"""
from dataExp import CorpusInterface as CI
from matplotlib import pyplot as plt 
import numpy as np
import os


"""
    These two functions can (and probably should) be rewritten as one function
"""

def author_text_length_plot(ci: CI, n_authors: int = 50):
    values = ci.get_authors_by_text_size()
    plt.rcParams["figure.figsize"] = (20,10)
    plt.title("Top " + str(n_authors) + " authors")
    x = []
    
    for i in range(n_authors):
        x.append(values[i][0])
        plt.bar(values[i][0],values[i][1], color = [np.random.rand(3) for i in range(n_authors)])
    plt.xticks(rotation='vertical')
    plt.xlabel("Authors")
    plt.ylabel("Number of words in all of author's texts")
    plt.savefig(os.getcwd()+"/Plots/author_word_plot.png")
    plt.close()
    return values, x

def author_work_count_plot(ci: CI, n_authors: int = 50):
    values = ci.get_authors_by_text_size(characterCount=False)
    plt.rcParams["figure.figsize"] = (20,10)
    plt.title("Top " + str(n_authors) + " authors")
    x = []

    for i in range(n_authors):
        x.append(values[i][0])
        plt.bar(values[i][0],values[i][1], color = [np.random.rand(3) for i in range(n_authors)])
    plt.xticks(rotation='vertical')
    plt.xlabel("Authors")
    plt.ylabel("Number of works per author")
    plt.savefig(os.getcwd()+"/Plots/author_work_count_plot.png")
    plt.close()
    return values

def author_lexical_diversity_plot(ci: CI, authors):
    values = ci.lexical_diversity(authors)
    plt.rcParams["figure.figsize"] = (20,10)
    plt.title("Top " + str(len(authors)) + " authors")
    x = []
    for i in range(len(authors)):
        x.append(values[i][0])
        plt.bar(values[i][0],values[i][1], color = [np.random.rand(3) for i in range(len(authors))])
    plt.xticks(rotation='vertical')
    plt.xlabel("Authors")
    plt.ylabel("Lexical diversity across all of author's texts")
    plt.savefig(os.getcwd()+"/Plots/author_lexical_diversity_plot.png")
    plt.close()


def all_plots():
    ci = CI()
    topAuthorsByTextLength, authors = author_text_length_plot(ci, n_authors=50)
    topAuthorsByWorkCount = author_work_count_plot(ci, n_authors=50)
    author_lexical_diversity_plot(ci, authors)

if __name__=="__main__":
    all_plots()