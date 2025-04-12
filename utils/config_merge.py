import dataclasses


def merge_configs(dataclass_cfg, dict_conf):
    """
    Recursively merge dictionary override config into dataclass config object,
    preserving defaults from dataclass.
    """
    for key, value in dict_conf.items():
        if hasattr(dataclass_cfg, key):
            sub_cfg = getattr(dataclass_cfg, key)
            if dataclasses.is_dataclass(sub_cfg) and isinstance(value, dict):
                merge_configs(sub_cfg, value)
            else:
                setattr(dataclass_cfg, key, value)
