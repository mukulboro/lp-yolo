# Simple Implementation of original Yolo v1 in PyTorch
## CS583 - Intro to Computer Vision 
## Michael Shenoda
## michael.shenoda@drexel.edu
---

## Install Requirements
pip install -r requirements.txt

# Detection
## Detect using tinier model
python .\detect.py -w yolo_tinier.pt -t tinier --conf 0.1 --iou 0.3

## Detect using tiny model
python .\detect.py -w yolo_tinier.pt -t tinier --conf 0.1 --iou 0.3

## Detect using ms model
python .\detect.py -w yolo_tinier.pt -t tinier --conf 0.1 --iou 0.3

# Training
## Train using tinier model
python .\train.py -t tinier

## Train using tiny model
python .\train.py -t tiny

## Train using ms model
python .\train.py -t ms


# Evaluation
## Evaluate tinier model
python .\evaluate.py -t tinier
## Evaluate tiny model
python .\evaluate.py -t tiny
## Evaluate ms model
python .\evaluate.py -t ms