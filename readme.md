
# Guide to using YOLOv2
This reference is mainly to help me and colleagues when we need to use real-time object detection in our work. I have experience with classification and localization using deep neural networks, but this is my first time to implement a deep, real-time detector + localizer since I normally detect based on cues such as motion, shape, color (thermal channel), etc.  

To learn about Joseph Redmond's YOLO:  
see his darknet website:  
https://pjreddie.com/darknet/yolo/  
and the project github page:  
https://github.com/pjreddie/darknet  
and read the YOLO9000 paper:  
https://arxiv.org/abs/1612.08242  

*** NOTE: this document is a work in progress... ***

I developed a code-crush (I'm coining that term unless it's already been used) on YOLOv2 when I discovered the repo and saw that it was written in C and CUDA. Maybe I was so enthralled because I had been suffering through hell with TensorFlow, which I only use because Keras is so great. Though I have been coding in Python for several years, I still love the beauty, power, and simplicity (until I get segfaults that take me an hour to debug) of the C programming language. What's more, calling C from Python is no problem. The darknet neural network framework is Open Source, fast, easy to compile and use, and supports both CPU and GPU.  


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

###### generate labels for VOC
```
~$ wget https://pjreddie.com/media/files/voc_label.py
~$ python voc_label.py
```

###### concatenate the text files
`~$ cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt`  

###### modify Cgf for Pascal data
Modify cfg/voc.data to point to your data.
```
  1 classes= 20
  2 train  = <path-to-voc>/train.txt
  3 valid  = <path-to-voc>2007_test.txt
  4 names = data/voc.names
  5 backup = backup
```

###### download the pre-trained convolutional weights
`~$ wget https://pjreddie.com/media/files/darknet19_448.conv.23`

###### Now train the model
```
~$ cd darknet
~$ mkdir backup
~$ ./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
Ctrl+C to stop.  
To resume training after stopping, copy the most recent weights file from backup to the darknet directory, say it's "yolo-voc_70000.weights", then:  
`~$ ./darknet detector train cfg/voc.data cfg/yolo-voc.cfg yolo-voc_70000.weights`  



### Training to detect custom objectd (your own dataset)

Nils Tijtgat created an informative post on the subject:  
https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/  
Guanghan Ning made his own detector for traffic signs:  
https://github.com/Guanghan/darknet  

The information below were shamelessly taken from Nils's post and resolved YOLOv2 GitHub issues. Below is a simple procedure to use YOLOv2 with your own dataset. For expanations and insights into the workings of YOLOv2, please read the YOLO9000 paper.  


###### Create the image and labels dataset
I tested with this annotated dataset:  
https://timebutt.github.io/content/other/NFPA_dataset.zip  

The annotations for this image set is alread in the YOLO version 2 format, but in the future, when I use "labelImg" to locate the training object bounding boxes, I will need to convert to the format expected by YOLOv2. This can be done with G. Ning's script found here: 
https://github.com/Guanghan/darknet/blob/master/scripts/convert.py

###### Create the train.txt and test.txt files
These files contain the paths of the images. Nils Tijtgat provides a script (https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/) to create these two files. Be aware that you can set the variable "percentage_test" to determine the percentage of images to be set aside for the test set.  

###### Create YOLOv2 configuration files
We must create three files. Create the first two like so:  
```
~$ touch cfg/nfpa.data
~$ touch cfg/obj.names
```

Then enter the values in each respective file:  

nfpa.data:  
```
classes= 1  
train  = /home/username/darknet/train.txt 
valid  = /home/username/darknet/test.txt
names  = /home/username/darknet/cfg/nfpa.names
backup = /home/username/darknet/backup/
```

nfpa.names:  
```
NFPA
```

Next, create the .cfg file. Copy the existing yolo-voc.2.0.cfg and edit it:  
```
~$ cd cfg
~$ cp yolo-voc.2.0.cfg yolo-nfpa.2.0.cfg
```  
Now make the following edits to yolo-nfpa.2.0.cfg:  
* Line 3: set batch=64, this means we will be using 64 images for every training step  
* Line 4: set subdivisions=8, the batch will be divided by 8 to decrease GPU VRAM requirements. If you have a powerful GPU with loads of VRAM, this number can be decreased, or batch could be increased. The training step will throw a CUDA out of memory error so you can adjust accordingly.  
* Line 244: set classes=1, the number of categories we want to detect  
* Line 237: set filters=(classes + 5)\*5 in our case filters=30  

##### Train
Train this data set just as we did for the VOC dataset above. Use the weights pre-trained on Imagenet (the file is darknet19_448.conv.23).  

There is one change that we should make to /examples/detector.c before training.  When the network is training, it will save the weights to /backup every 100 iterations until 900. After 900 iterations, the default setting is to save every 10,000 iterations. We would like to save more often that that with this small dataset. To change this setting, change the following line in examples/detector.c  
`if(i%10000==0 || (i < 1000 && i%100 == 0)){`  
to  
`if(i%100==0 || (i < 1000 && i%100 == 0)){`  
or simply replace the number 10000 with whatever smaller whole number you want.

Now train the nework:  
```
~$ ./darknet detector train cfg/nfpa.data cfg/yolo-nfpa.2.0.cfg darknet19_448.conv.23
```  


##### Test (using our trained network to detect our class of interest)  
To detect our class in a test image, say "data/testimage.jpg", copy the best weights file, say it's "yolo-obj_1000.weights" to darknet directory. Then:  
```
~$ ./darknet detector test cfg/obj.data cfg/yolo-obj.cfg yolo-obj1000.weights data/testimage.jpg
```  
      