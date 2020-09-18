#!/usr/bin/env python3
"""Unigram BLEU score"""
import numpy as np


def compute_presicion(references, sentence, n):
    win_size = n
    tokenized_sentence = []
    word_instaces = {}
    is_there = False
    count = 0
    clip_list = []

    # split the sentence on sublist of windows size
    for idx in range(len(sentence) - win_size + 1):
        tokenized_sentence.append(sentence[idx:win_size + idx])

    # count the unigrams on sentence
    for token in tokenized_sentence:
        for idx in range(len(sentence) - win_size + 1):
            if token == sentence[idx:idx + win_size]:
                count += 1

    # count the cliping on references
    for ref in references:
        clip = 0
        for idx in range(len(ref) - win_size + 1):
            for token in tokenized_sentence:
                if ref[idx:idx + win_size] == token:
                    clip += 1
        clip_list.append(clip)
    final_clip = max(clip_list)
    return final_clip / count


def cumulative_bleu(references, sentence, n):
    """
    calculates the cumulative n-gram BLEU score for a sentence:
    :param references: is a list of reference translations
    :param sentence: is a list containing the model proposed sentence
    :return:
    """
    precisions = [0] * n
    for idx in range(0, n):
        precisions[idx] = compute_presicion(references, sentence, idx+1)

    geo_mean = np.exp(np.sum(np.log(precisions) / n))

    len_trans = len(sentence)

    # Brevity penalty
    # closest reference length from translation length
    closest_ref_idx = np.argmin([abs(len(x) - len_trans) for x in references])
    reference_length = len(references[closest_ref_idx])

    if len_trans > reference_length:
        BP = 1
    else:
        BP = np.exp(1 - float(reference_length) / len_trans)

    bleu = BP * geo_mean

    return bleu
