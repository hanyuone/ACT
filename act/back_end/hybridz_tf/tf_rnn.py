from act.back_end.core import Bounds, Fact
from act.back_end.interval_tf.tf_rnn import (
    tf_lstm, tf_gru, tf_rnn, tf_embedding,
)


def hz_tf_lstm(L, bounds, tf):
    return tf_lstm(L, bounds)


def hz_tf_gru(L, bounds, tf):
    return tf_gru(L, bounds)


def hz_tf_rnn(L, bounds, tf):
    return tf_rnn(L, bounds)


def hz_tf_embedding(L, bounds, tf):
    return tf_embedding(L, bounds)
