import torch
import torch.nn as nn
import sys
sys.path.append('core')
from core.Model import build_flowmodel
from configs.train_config.SAMFlow_H_train_things import get_cfg

cfg = get_cfg()
cfg.sam_checkpoint = 'sam_vit_h_4b8983.pth'
model = build_flowmodel(cfg)
state_dict = model.state_dict()

#Load Flowformer checkpoint
ff_state_dict = torch.load('flowformer-things.pth', map_location='cpu')


for k in state_dict.keys():
    if k in ff_state_dict:
        state_dict[k] = ff_state_dict[k].clone()
    elif 'module.' + k in ff_state_dict:
        state_dict[k] = ff_state_dict['module.' + k].clone()
        # print(k)

model.load_state_dict(state_dict, strict=True)

torch.save(model.state_dict(), 'samflow-init-checkpoint.pth')