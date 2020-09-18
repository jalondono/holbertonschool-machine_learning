#!/usr/bin/env python3
"""Unigram BLEU score"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    calculates the cumulative n-gram BLEU score for a sentence:
    :param references: is a list of reference translations
    :param sentence: is a list containing the model proposed sentence
    :return:
    """
