#!/usr/bin/env bash

gpus=0,1
checkpoint_root=checkpoints
data_name=All

img_size=256
batch_size=8
lr=0.01
max_epochs=200
net_G=DIELNet
lr_policy=linear

split=train_multi
split_val=val_multi
project_name=${net_G}_${data_name}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}