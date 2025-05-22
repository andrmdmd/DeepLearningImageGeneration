from .train_engine import Engine
from .sweep_engine import SweepEngine
from .unet2d_engine import UNet2DEngine

def build_engine(engine_name: str):
    if engine_name == "engine":
        return Engine
    elif engine_name == "sweep_engine":
        return SweepEngine
    elif engine_name == "unet2d_engine":
        return UNet2DEngine
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
