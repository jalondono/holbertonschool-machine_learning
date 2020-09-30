#!/usr/bin/env python3
"""Dataset class"""

import tensorflow_datasets as tfds


class Dataset:
    """ Class """
    def __init__(self):
        """
        Constructor method
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                       as_supervised=True)
        train_examples, val_examples = examples['train'], examples['validation']
        self.data_train = train_examples
        self.data_valid = val_examples
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

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
