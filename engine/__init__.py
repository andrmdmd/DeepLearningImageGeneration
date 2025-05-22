from .train_engine import Engine
from .sweep_engine import SweepEngine
from .unet2d_engine import UNet2DEngine
from .dcgan_engine import DCGANEngine

def build_engine(engine_name: str, is_sweep: bool = False):
    engine_class = None
    if engine_name == "engine": # probably will be removed
        engine_class =  Engine
    elif engine_name == "unet2d_engine":
        engine_class =  UNet2DEngine
    elif engine_name == "dcgan_engine":
        return DCGANEngine
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
    
    if is_sweep:
        return lambda accelerator, cfg: SweepEngine(engine_class, accelerator, cfg)
