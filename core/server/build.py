from ..utils import Registry

SERVER_REGISTRY = Registry("service_type")
def build_server(cfg, *args, **kwargs):
    name = cfg.pop("DETECTION_TYPE")
    server = SERVER_REGISTRY.get(name)(cfg.pop("config_path"), *args, **kwargs)
    return server