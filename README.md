## YOLO_tensorflow

Tensorflow implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf), including training and test phase.

#### 这个版本我加了注释，按照下面的方法安装后运行test.py即可实现实时目标检测

#### 如果帮到了您，可以点击右上角给我一个star~

### Installation

1. Clone yolo_tensorflow repository
	```Shell
	$ git clone https://github.com/taogelose/my_yolo_tensorflow.git
    $ cd yolo_tensorflow
	```

2. Download Pascal VOC dataset, and create correct directories （下载Pascal VOC数据集，下面方法较慢的话，建议上网找
	```Shell
	$ ./download_data.sh
	```


3. Download [YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing)
weight file and put it in `data/weight`  （这个YOLO small是训练好了的参数，谷歌无法访问可以在百度下载，或者找我要。

4. Modify configuration in `yolo/config.py`  （这个是参数列表，建议不要动

5. Training
   （没有好卡建议不要train
	```Shell
	$ python train.py
	```


6. Test
   （test里面有两种test方法，一种是用电脑摄像头，一种是通过路径输入图片。代码中后一种被注释掉了。
	```Shell
	$ python test.py
	```


### Requirements
1. Tensorflow

2. OpenCV
