import torch
def build_flowmodel(cfg):
    name = cfg.transformer 
    if name in ['SAMFlow_H', 'SAMFlow_B', 'SAMFlow_tiny']:
        from .LatentCostFormer.net_SAMFlow import FlowModel
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")
    return FlowModel(cfg.FlowModel)