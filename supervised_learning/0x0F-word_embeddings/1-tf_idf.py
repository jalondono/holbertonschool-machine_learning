#!/usr/bin/env python3
"""Bag of words"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding:
    :param sentences: is a list of sentences to analyze
    :param vocab: is a list of the vcabulary words to use for the analysis
    :return: embeddings, features
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer.get_feature_names()
