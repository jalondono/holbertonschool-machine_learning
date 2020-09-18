#!/usr/bin/env python3
"""Unigram BLEU score"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    calculates the n-gram BLEU score for a sentence:
    :param references: is a list of reference translations
    :param sentence: is a list containing the model proposed sentence
    :return:
    """
    #  initialization of variables
    win_size = 1
    tokenized_sentence = []
    word_instaces = {}
    is_there = False
    count = 0
    clip_list = []

    # split the sentence on sublist of windows size
    for idx in range(len(sentence) - win_size + 1):
        tokenized_sentence.append(sentence[idx:win_size + idx])
    print()

    # count the unigrams on sentence
    for token in tokenized_sentence:
        for idx in range(len(sentence) - win_size + 1):
            if token == sentence[idx:idx + win_size]:
                count += 1

    # min len of references
    r_list = np.array([np.abs(len(s) - count) for s in references])
    r_ind = np.argwhere(r_list == np.min(r_list))
    lens = np.array([len(s) for s in references])[r_ind]
    r = np.min(lens)

    # count the cliping on references
    for ref in references:
        clip = 0
        for idx in range(len(ref) - win_size + 1):
            for token in tokenized_sentence:
                if ref[idx:idx + win_size] == token:
                    clip += 1
        clip_list.append(clip)
    final_clip = max(clip_list)
    precision = final_clip / count

    if count > r:
        BP = 1
    else:
        BP = np.exp(1 - (r / count))
        Bleu = BP * precision
    return Bleu
