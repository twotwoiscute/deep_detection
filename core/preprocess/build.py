from ..utils import Registry 

PREPROCESSOR_REGISTRY = Registry("preprocess")

def build_preprocessor(cfg, *args, **kwargs):
    name = cfg.get("name")
    preprocessor = PREPROCESSOR_REGISTRY.get(name)(cfg, *args, **kwargs)
    return preprocessor