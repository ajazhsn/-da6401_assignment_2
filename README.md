# DA6401 Assignment 2 — Visual Perception Pipeline

## GitHub Repository
https://github.com/ajazhsn/-da6401_assignment_2

## Public W&B Report
[YOUR_WANDB_REPORT_LINK_HERE](https://wandb.ai/zainjimess-iitmaana/da6401-a2/reports/Introduction-to-Deep-Learning---VmlldzoxNjQ4NDY5OA?accessToken=mnkies5tir4hy22xq1ww3ix0hm9s7xjmhflwf6tmc7897m2vpn755ry49x64z2dk)

## Setup
pip install -r requirements.txt

## Training
python train.py --task classify --data_root oxford-iiit-pet --epochs 60
python train.py --task localize --data_root oxford-iiit-pet --epochs 40
python train.py --task segment  --data_root oxford-iiit-pet --epochs 30

## Inference
python inference.py --image path/to/image.jpg
