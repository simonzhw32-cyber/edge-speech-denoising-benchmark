"""Explicit profiles; the original six-layer default is preserved."""

from types import MappingProxyType

from .model import TFGridNetModel

DNS_PROFILE = "dns_ins20_epoch33"
DNS_MODEL_PARAMS = MappingProxyType({
    "n_fft": 512, "stride": 256, "n_layers": 4,
    "lstm_hidden_units": 128, "attn_n_head": 4,
    "attn_approx_qk_dim": 512, "emb_dim": 32,
    "emb_ks": 4, "emb_hs": 4, "activation": "prelu", "eps": 1e-5,
})
DNS_PARAMETERS = 2552790
DNS_STATE_ENTRIES = 362


def build_tfgridnet_dns_model():
    """Return a random CPU FP32 model; weights require explicit loading."""
    return TFGridNetModel(**dict(DNS_MODEL_PARAMS))
