import sys
import os
import re
import operator
import warnings
import argparse
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.NOTSET)

import wandb
import gensim
import pandas as pd

from gensim.models import CoherenceModel, LdaModel, LdaMulticore
from gensim.models.callbacks import CoherenceMetric, ConvergenceMetric, Callback
from gensim.corpora import Dictionary
from pprint import pprint


# class ConvergenceCallback(Callback):

#     def __init__(self):
#         super().__init__([ConvergenceMetric()])
#         self.epoch = 0
#         self.logger = None

#     def get_value(self, **kwargs):
#         with open('test.log', 'w+') as f:
#             f.write(model.alpha)
#             f.write(',')
#             f.write(model.eta)
#             f.write('\n')
#         self.epoch += 1



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


def evaluate_lda_models(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
    
    return lm_list, c_v


def ret_top_model(texts, corpus, dictionary):
    """
    Since LDA is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.6:
        lm = LdaMulticore(corpus=corpus, id2word=dictionary, workers=3, num_topics=9)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
        print(top_topics[0])
    return lm, top_topics


def save_objects():
    train_texts = list(build_texts('feedback.csv'))
    bigram = gensim.models.Phrases(train_texts)  # for bigram collocation detection
    bigram.save(os.path.join('model', 'bigram.pkl'))

    train_texts = process_texts(train_texts, bigram)
    with open(os.path.join('model', 'texts.txt'), 'w') as f:
        for doc in train_texts:
            f.write(','.join(doc) + '\n')

    dictionary = Dictionary(train_texts)
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


def get_default_hyperparameters():
    return dict(
        passes = 20,
        iterations = 400,
        decay = 0.5,
        offset = 1,
        chunksize = 2000,
        alpha='auto',
        eta='auto',
        random_state = 123,
        num_topics = 6,
        no_below=5,
        no_above=0.5
    )


def get_kwargs(argv):
    kwargs = dict()
    for arg in argv[1:]:
        name, value = arg.strip('--').split('=')
        kwargs[name] = value
    return kwargs


def train(passes, iterations, num_topics, decay, offset, chunksize, alpha, eta, random_state, no_below, no_above):
    texts, _, dictionary = get_text_inputs_from_folder('model')

    dictionary.filter_extremes(no_above=no_above, no_below=no_below)
    corpus = [dictionary.doc2bow(text) for text in texts]

    convergence_logger = ConvergenceMetric(logger='shell')
    coherence_logger = CoherenceMetric(logger='shell', corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v')

    lm = LdaModel(
        corpus=corpus, 
        id2word=dictionary, 
        eval_every=1, 
        callbacks=[convergence_logger, coherence_logger],
        passes=passes,
        iterations=iterations,
        num_topics=num_topics,
        decay=decay,
        offset=offset,
        chunksize=chunksize,
        alpha=alpha,
        eta=eta,
        random_state=random_state
    )
    
    lm.save(os.path.join(wandb.run.dir, f'lda_{num_topics}t.model'))

    cm = CoherenceModel(
        model=lm, 
        texts=texts, 
        coherence='c_v')

    coherence = cm.get_coherence()
    wandb.log({'coherence': coherence})


if __name__ == "__main__":
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
    args = parser.parse_args()

    hyperparameters = vars(args)
    wandb.init(project="bom-topic-modelling", config=hyperparameters)
    train(**hyperparameters)