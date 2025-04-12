from .train_engine import Engine
from .sweep_engine import SweepEngine


def build_engine(engine_name: str):
    if engine_name == "engine":
        return Engine
    elif engine_name == "sweep_engine":
        return SweepEngine
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
