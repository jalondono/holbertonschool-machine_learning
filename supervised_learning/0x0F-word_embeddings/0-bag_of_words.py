#!/usr/bin/env python3
"""Bag of words"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix:
    :param sentences: is a list of sentences to analyze
    :param vocab: is a list of the vocabulary words to use for the analysis
    :return: embeddings, features
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer.get_feature_names()
