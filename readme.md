
# Guide to using YOLOv2
This reference is mainly to help me and colleagues when we need to use real-time object detection in our work. I have experience with classification and localization using deep neural networks, but have yet to implement a deep, real-time detector + localizer since I normally detect based on cues such as motion, shape, color (thermal channel), etc.  

To learn about Joseph Redmond's YOLO:  
see his darknet website:  
https://pjreddie.com/darknet/yolo/  
and the project github page:  
https://github.com/pjreddie/darknet  
and read the YOLO9000 paper:  
https://arxiv.org/abs/1612.08242  

*** NOTE: this document is a work in progress... ***

I developed a code-crush (I'm coining that term unless it's already taken) on YOLOv2 when I discovered the repo and saw that it was written in C and CUDA. Maybe I was so enthralled because I had been suffering through hell with TensorFlow, which I only use because Keras is so great. Though I have been coding in Python for many years, I still love the beauty, power, and simplicity (until I get segfaults that take me an hour to debug) of the C programming language. What's more, calling C from Python is no problem. The darknet neural network framework is Open Source, fast, easy to compile and use, and supports both CPU and GPU.  


### Training YOLO on VOC
###### get the Pascal VOC data
```
~$ cd darknet/data
~$ mkdir voc
~$ cd voc
~$ wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
~$ wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
~$ wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
~$ tar xf VOCtrainval_11-May-2012.tar
~$ tar xf VOCtrainval_06-Nov-2007.tar
~$ tar xf VOCtest_06-Nov-2007.tar
```
There will now be a VOCdevkit/ subdirectory with all the VOC training data in it.

##### generate labels for VOC
```
~$ wget https://pjreddie.com/media/files/voc_label.py
~$ python voc_label.py
```

##### concatenate the text files
`~$ cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt`  

##### modify Cgf for Pascal data
Modify cfg/voc.data to point to your data.
```
  1 classes= 20
  2 train  = <path-to-voc>/train.txt
  3 valid  = <path-to-voc>2007_test.txt
  4 names = data/voc.names
  5 backup = backup
```

##### download the pre-trained convolutional weights
`~$ wget https://pjreddie.com/media/files/darknet19_448.conv.23`

##### Now train the model
```
~$ cd darknet
~$ mkdir backup
~$ ./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
Ctrl+C to stop.  
To resume training after stopping, copy the most recent weights file from backup to the darknet directory, say it's "yolo-voc_70000.weights", then:  
`~$ ./darknet detector train cfg/voc.data cfg/yolo-voc.cfg yolo-voc_70000.weights`  



### Training to detect custom objectd (your own dataset)
folow the advice of Alexey:  
https://github.com/AlexeyAB/darknet  
( I have cloned the repo in ~/Acq_and_Track/ )  
Nils Tijtgat created an informative post on the subject:  
https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/  


##### Create the image and labels dataset
I tested with this annotated dataset:  
https://timebutt.github.io/content/other/NFPA_dataset.zip  


##### Create the train.txt and test.txt files
These files contain the paths of the images. I'll create a script to make these.  

##### Create YOLOv2 configuration files
We must create three files:  
* cfg/obj.data
* cfg/obj.names
* cfg/yolo-obj.cfg

Yolov2.cfg:
```
classes= 1  
train  = train.txt  
valid  = test.txt  
names = obj.names  
backup = backup/
```

obj.names:
```
NFPA
```

yolo-obj.cfg:
```
~$ cd cfg
~$ cp yolo-voc.cfg yolo-obj.cfg
```
Now make the following edits to yolo-obj.cfg:
* Line 3: set batch=64, this means we will be using 64 images for every training step
* Line 4: set subdivisions=8, the batch will be divided by 8 to decrease GPU VRAM requirements. If you have a powerful GPU with loads of VRAM, this number can be decreased, or batch could be increased. The training step will throw a CUDA out of memory error so you can adjust accordingly.
* Line 244: set classes=1, the number of categories we want to detect
* Line 237: set filters=(classes + 5)*5 in our case filters=30

##### Train
Train this data set just as we did for the VOC dataset above. Use the weights pre-trained on Imagenet just as we did above for the VOC dataset (the file is darknet19_448.conv.23).

##### Test (using our trained network to detect our class of interest)  
To detect our class in a test image, say "data/testimage.jpg", copy the weights file, say it's "yolo-obj_1000.weights" to darknet directory. Then:  
`~$ ./darknet detector test cfg/obj.data cfg/yolo-obj.cfg yolo-obj1000.weights data/testimage.jpg`  
