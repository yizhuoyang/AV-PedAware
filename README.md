# AV-PedAware

This is the repository for "AV-PedAware: Self-Supervised Audio-Visual Fusion for Dynamic Pedestrian Awareness"

<img src="https://github.com/yizhuoyang/AV-PedAware/blob/main/figs/detection_result.gif" width="70%">

## Data
Some of the newly collected data with 8 mic array can be download from this [link](https://pan.baidu.com/s/1VzQnecSW_UPeBkFju6Zf9A?pwd=2024)

## Train
```bash
$ python train.py --train_epoch 200 --workers 4 --gpu cuda:0
```
## Evaluate
```bash
$ python evaluation.py --checkpoint_path /model_path --gpu cuda:0 
```
## Tutorial
A tutorial is added to show the general workflow of the network with detailed explanation.

## Cite
```bash
@inproceedings{yang2023av,
  title={AV-PedAware: Self-Supervised Audio-Visual Fusion for Dynamic Pedestrian Awareness},
  author={Yang, Yizhuo and Yuan, Shenghai and Cao, Muqing and Yang, Jianfei and Xie, Lihua},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1871--1877},
  year={2023},
  organization={IEEE}
}
```
