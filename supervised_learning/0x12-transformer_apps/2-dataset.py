#!/usr/bin/env python3
"""Dataset class"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ dataset class"""
    def __init__(self):
        """
        Constructor method
        """
        exam, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True)
        train_examples, val_examples = exam['train'], exam['validation']
        self.data_train = train_examples.map(self.tf_encode)
        self.data_valid = val_examples.map(self.tf_encode)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            train_examples)

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset:
        :param data: is a tf.data.Dataset whose examples are
         formatted as a tuple (pt, en)
        :return: tokenizer_pt, tokenizer_en
        """
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
         encodes a translation into tokens:
        :param pt: is the tf.Tensor containing the Portuguese sentence
        :param en: is the tf.Tensor containing the corresponding
        English sentence
        :return: pt_tokens, en_tokens
        """
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return lang1, lang2

    def tf_encode(self, pt, en):
        """
        acts as a tensorflow wrapper for the encode instance method
        :param pt:
        :param en:
        :return:
        """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
