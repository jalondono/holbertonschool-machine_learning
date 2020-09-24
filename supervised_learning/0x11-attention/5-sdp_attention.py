#!/usr/bin/env python3
""" Scaled Dot Product Attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    calculates the scaled dot product attention:
    :param Q: is a tensor with its last two dimensions as
    (..., seq_len_q, dk) containing the query matrix
    :param K: is a tensor with its last two dimensions as
     (..., seq_len_v, dk) containing the key matrix
    :param V: is a tensor with its last two dimensions as
    (..., seq_len_v, dv) containing the value matrix
    :param mask: is a tensor that can be broadcast into
    (..., seq_len_q, seq_len_v) containing the optional
     mask, or defaulted to None
    :return: output, weights
    """
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

    return output, attention_weights
