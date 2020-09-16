#!/usr/bin/env python3
""" Train Word2Vec """
from gensim.models import Word2Vec
from gensim.test.utils import common_texts


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    creates and trains a gensim word2vec model:
    :param sentences: is a list of sentences to be trained on
    :param size: is the dimensionality of the embedding layer
    :param min_count: is the minimum number of occurrences of a word
     for use in training
    :param window: is the maximum distance between the current
    and predicted word within a sentence
    :param negative: is the size of negative sampling
    :param cbow: is a boolean to determine the training type;
     True is for CBOW; False is for Skip-gram
    :param iterations: is the number of iterations to train over
    :param seed: is the seed for the random number generator
    :param workers: is the number of wrker threads to train the model
    :return: the trained model
    """
    model = Word2Vec(sentences, size=size, window=window,
                     min_count=min_count, negative=negative,
                     sg=cbow, seed=seed, iter=iterations)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
