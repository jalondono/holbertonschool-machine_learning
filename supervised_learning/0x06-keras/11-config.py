#!/usr/bin/env python3
"""  Save and Load Configuration """
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format:
    :param network:
    :param filename:
    :return:
    """
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)
    return None


def load_config(filename):
    """
    loads a model with a specific configuration:
    :param filename:
    :return:
    """
    with open(filename) as json_file:
        json_config = json_file.read()
    new_model = K.models.model_from_json(json_config)
    return new_model
