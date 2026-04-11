from act.back_end.core import Bounds, Fact
from act.back_end.interval_tf.tf_transformer import (
    tf_posenc, tf_layernorm, tf_gelu, tf_att_scores,
    tf_softmax, tf_att_mix, tf_mha_split, tf_mha_join, tf_mask_add,
    tf_embedding,
)


def hz_tf_posenc(L, bounds, tf):
    return tf_posenc(L, bounds)


def hz_tf_layernorm(L, bounds, tf):
    return tf_layernorm(L, bounds)


def hz_tf_gelu(L, bounds, tf):
    return tf_gelu(L, bounds)


def hz_tf_att_scores(L, bounds, tf):
    return tf_att_scores(L,
        tf._before[L.params["q_src"]].bounds,
        tf._before[L.params["k_src"]].bounds)


def hz_tf_softmax(L, bounds, tf):
    return tf_softmax(L, bounds)


def hz_tf_att_mix(L, bounds, tf):
    return tf_att_mix(L,
        tf._before[L.params["w_src"]].bounds,
        tf._before[L.params["v_src"]].bounds)


def hz_tf_mha_split(L, bounds, tf):
    return tf_mha_split(L, bounds)


def hz_tf_mha_join(L, bounds, tf):
    return tf_mha_join(L,
        tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before))


def hz_tf_mask_add(L, bounds, tf):
    return tf_mask_add(L, bounds)
