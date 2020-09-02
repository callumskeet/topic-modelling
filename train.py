import sys
import os
import re
import argparse
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)

import wandb
import pandas as pd

import gensim
from gensim.models import LdaModel
from gensim.models.callbacks import CoherenceMetric, ConvergenceMetric, PerplexityMetric, DiffMetric
from gensim.corpora import Dictionary

import pyLDAvis
import pyLDAvis.gensim
import plotly.express as px


class ConvergenceCallback(ConvergenceMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch = 0

    def get_value(self, **kwargs):
        value = super().get_value(**kwargs)
        wandb.log({'convergence': value}, step=self.epoch)
        self.epoch += 1
        return value


class CoherenceCallback(CoherenceMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch = 0

    def get_value(self, **kwargs):
        value = super().get_value(**kwargs)
        wandb.log({'coherence': value}, step=self.epoch)
        self.epoch += 1
        return value


class PerplexityCallback(PerplexityMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch = 0

    def get_value(self, **kwargs):
        value = super().get_value(**kwargs)
        wandb.log({'perplexity': value}, step=self.epoch)
        self.epoch += 1
        return value


class DiffCallback(DiffMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch = 0

    def get_value(self, **kwargs):
        value = super().get_value(**kwargs)
        wandb.log({'topic_diff': value}, step=self.epoch)
        self.epoch += 1
        return value


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


def get_hyperparameters(passes=20, iterations=400, decay=0.5, offset=1024, chunksize=2000, alpha='auto', 
                        eta='auto', random_state=1024, num_topics=7, no_below=None, no_above=None):
    return dict(
        passes = passes,
        iterations = iterations,
        decay = decay,
        offset = offset,
        chunksize = chunksize,
        alpha = alpha,
        eta = eta,
        random_state = random_state,
        num_topics = num_topics,
        no_below = no_below,
        no_above = no_above
    )


def train(passes=1, iterations=50, num_topics=100, decay=0.5, offset=1.0, chunksize=2000, 
          alpha='symmetric', eta=None, random_state=None, no_below=None, no_above=None):
    
    texts, _, dictionary = get_text_inputs_from_folder('model')

    if no_above or no_below:
        dictionary.filter_extremes(no_above=no_above, no_below=no_below)
    corpus = [dictionary.doc2bow(text) for text in texts]

    convergence_logger = ConvergenceCallback(logger='shell')
    coherence_logger = CoherenceCallback(logger='shell', corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v')
    perplexity_logger = PerplexityCallback(logger='shell', corpus=corpus)

    lm = LdaModel(
        corpus=corpus, 
        id2word=dictionary, 
        eval_every=1, 
        callbacks=[convergence_logger, coherence_logger, perplexity_logger],
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

    return lm, corpus, dictionary


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
    return parser.parse_args()


def main():
    hyperparameters = get_hyperparameters()
    if len(sys.argv) > 1:
        args = vars(parse_args())
        args = {k: v for k, v in args.items() if v is not None}
        hyperparameters.update(args)

    wandb.init(project="bom-topic-modelling", config=hyperparameters)

    lm, corpus, dictionary = train(**hyperparameters)

    lm.save(os.path.join(wandb.run.dir, 'lda.model'))
    
    # topic difference heatmap
    mdiff, _ = lm.diff(lm, distance='jaccard', num_words=50)
    fig = px.imshow(mdiff, origin='lower', color_continuous_scale='RdBu_r')
    wandb.log({"topic_diff": fig})
    
    # pyLDAvis
    vis = pyLDAvis.gensim.prepare(lm, corpus, dictionary)
    html = pyLDAvis.prepared_data_to_html(vis)
    wandb.log({"pyLDAvis": wandb.Html(html, inject=False)})


if __name__ == "__main__":
    main()