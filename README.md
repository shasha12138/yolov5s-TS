### Install

Clone repo and install requirements.txt in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```
git clone https://github.com/shasha12138/yolov5s-TS  # clone
cd yolov5s-TS
pip install -r requirements.txt  # install
```



### Detect

```
python detect.py --weights yolov5s.pt --source img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
```



### Training

1、we use Yolo format for training. Start by creating a folder to store the dataset

2、Create a .yaml file that corresponds to the dataset being used.

3、start network training.

```
python train.py --data yours.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
```



### Val

**mAP**

```
python val.py --data yolov5n.yaml --img 640 --conf 0.001 --iou 0.65
```

**Speed**

```
python val.py --data coco.yaml --img 640 --task speed --batch 1
```

**TTA**

```
python val.py --data coco.yaml --img 1536 --iou 0.7 --augment
```

