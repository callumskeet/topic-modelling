import os
import argparse

import pandas as pd

import gensim
from gensim.corpora import Dictionary


def build_texts(fname):
    """
    Function to build tokenized texts from file
    
    Parameters:
    ----------
    fname: File to be read
    
    Returns:
    -------
    yields preprocessed line
    """
    df = pd.read_csv(fname)
    data = df['Brief Description']
    data = data.dropna()
    mask = data.str.contains(r'^no comment$|^comment not applicable')
    data = data[~mask]
    data = data.unique()
    for line in data:
        yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)


def process_texts(texts, ngram_model=None, stop_words=None):
    """
    Function to process texts. The following steps are taken:
    
    1. Collocation detection.
    2. Stopword removal.
    
    Parameters:
    ----------
    texts: Tokenized texts
    ngram_model: gensim Phrases model to generate n_grams from texts
    remove_stop_words: boolean value
    
    Returns:
    -------
    texts: Pre-processed tokenized texts
    """
    if ngram_model:
        texts = [ngram_model[line] for line in texts]
    if stop_words:
        texts = [[word for word in line if word not in stop_words] for line in texts]
    return texts


def save_objects():
    train_texts = list(build_texts('feedback.csv'))
    bigram = gensim.models.Phrases(train_texts)  # for bigram collocation detection
    bigram.save(os.path.join('model', 'bigram.pkl'))

    train_texts = process_texts(train_texts, bigram)
    with open(os.path.join('model', 'texts.txt'), 'w') as f:
        for doc in train_texts:
            f.write(','.join(doc) + '\n')

    dictionary = gensim.corpora.Dictionary(train_texts)
    dictionary.save(os.path.join('model', 'dictionary.pkl'))


def get_text_inputs_from_folder(directory):
    """
    Retrieves documents, bigram and dictionary from the specified directory.

    Parameters:
    ----------
    directory: The directory containing the text inputs. 
               Expects the following files: texts.txt, bigram.pkl, dictionary.pkl
    
    Returns:
    -------
    train_texts
    bigram
    dictionary
    corpus
    """
    with open(os.path.join(directory, 'texts.txt'), 'r') as f:
        train_texts = [line.strip().split(',') for line in f.readlines()]
        
    bigram = gensim.models.Phrases.load(os.path.join('model', 'bigram.pkl'))
    train_texts = process_texts(train_texts, bigram)

    dictionary = Dictionary.load(os.path.join('model', 'dictionary.pkl'))
    return train_texts, bigram, dictionary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=str)
    parser.add_argument("--chunksize", type=int)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--eta", type=str)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--no_above", type=float)
    parser.add_argument("--no_below", type=int)
    parser.add_argument("--num_topics", type=int)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--passes", type=int)
    parser.add_argument("--random_state", type=int)
    parser.add_argument("--minimum_probability", type=float)
    return parser.parse_args()