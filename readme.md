
# Guide to using YOLOv2
This reference is mainly to help me and colleagues when we need to use real-time object detection in our work. I have experience with classification and localization using deep neural networks, but this is my first time to implement a deep, real-time detector + localizer since I normally detect based on cues such as motion, shape, color (thermal channel), etc.  

To learn about Joseph Redmon's YOLO:  
see his darknet website:  
https://pjreddie.com/darknet/yolo/  
and the project github page:  
https://github.com/pjreddie/darknet  
and read the YOLO9000 paper:  
https://arxiv.org/abs/1612.08242  


I developed a code-crush (I'm coining that term if it's not already being used) on YOLOv2 when I discovered the repo and saw that it was written in C and CUDA. Maybe I was so enthralled because I had been suffering through hell with TensorFlow, which I only use because Keras is so great. Though I have been coding in Python for several years, I still love the beauty, power, and simplicity (until I get segfaults that take me an hour to debug) of the C programming language. The darknet neural network framework is Open Source, fast, easy to compile and use, and supports both CPU and GPU.  


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



### Training to detect custom objects (your own dataset)

Nils Tijtgat created an informative post on the subject:  
https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/  

Some of the information below was shamelessly taken from Nils's post and resolved YOLOv2 GitHub issues. Below is a simple procedure to use YOLOv2 with your own dataset. For explanations and insights into the workings of YOLOv2, please read the YOLO9000 paper.  


###### Create the image and labels dataset
I tested with this annotated dataset:  
https://timebutt.github.io/content/other/NFPA_dataset.zip  

The annotations for this image set is already in the YOLO version 2 format, but in the future, when I use "labelImg" to locate the training object bounding boxes, I will need to convert to the format expected by YOLOv2. This can be done with G. Ning's script found here: 
https://github.com/Guanghan/darknet/blob/master/scripts/convert.py  
  

###### Create the train.txt and test.txt files
These files contain the paths of the images. Nils Tijtgat provides a script (https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/) to create these two files. Be aware that you can set the variable "percentage_test" to determine the percentage of images to be set aside for the test set.  

If I remember correctly, one of the images in the NFPA dataset had an extension of ".JPEG", instead of ".jpg". Be sure to change this, or either make your script account for differing file extensions.

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

There is one change that we should make to examples/detector.c before training.  When the network is training, it will save the weights to the directory "backup" every 100 iterations until 900. After 900 iterations, the default setting is to save every 10,000 iterations. We would like to save more often than that with this small dataset. To change this setting, change the following line in examples/detector.c  
`if(i%10000==0 || (i < 1000 && i%100 == 0)){`  
to  
`if(i%100==0 || (i < 1000 && i%100 == 0)){`  
or simply replace the number 10000 with whatever smaller integer you want.

Now train the network:  
```
~$ ./darknet detector train cfg/nfpa.data cfg/yolo-nfpa.2.0.cfg darknet19_448.conv.23
```  


##### Test (using our trained network to detect our class of interest in a heretofore unseen image)  
To detect our class in a test image, say "data/testimage.jpg", copy the best weights file, say it's "yolo-nfpa_1500.weights", to darknet directory. Then:  
```
~$ ./darknet detector test cfg/nfpa.data cfg/yolo-nfpa.2.0.cfg yolo-nfpa_1500.weights data/testimage.jpg
```  

### Detection from Python  
My preferred way to invoke the trained YOLO model from Python is to use pyyolo (https://github.com/digitalbrain79/pyyolo). Honestly, after attempting the darknet python lib and failing and after making a clumsy script using subprocess, I think you'd agree that pyyolo is pretty neat.  
In Makefile, set GPU and CUDNN to 1 if you want to use your GPU, otherwise set these to 0. Likewise, in the instructions below, you can choose between setup_gpu.py and setup.py.  
Following the instructions in the readme:
```
~$ git clone --recursive https://github.com/thomaspark-pkj/pyyolo.git
~$ make
~$ rm -rf build
~$ python3 setup_gpu.py build
~$ python3 setup_gpu.py install
```
Then run the provided test script  
`python3 example.py`  

Using you own trained weights is as easy as copying some files from you darknet directory to your pyyolo directory and then making some few changes to example.py. Here is how I did it:

Copy nfpa.data and yolo-nfpa.2.0.cfg from the original darknet/cfg directory to ~/pyyolo/darknet/cfg

Copy yolo-nfpa_1500.weights from the original darknet directory to ~/pyyolo

Copy the test images, in my case fd_01.jpg, fd_02.jpg, and fd_03.jpg, to ~/pyyolo/data

Modify example.py appropriately. Here is my script:

```
import pyyolo
import numpy as np
import sys
from cv2 import imread

darknet_path = './darknet'
datacfg = 'cfg/nfpa.data'
cfgfile = 'cfg/yolo-nfpa.2.0.cfg'
weightfile = '../yolo-nfpa_1500.weights'
files = [ darknet_path + '/data/fd_01.jpg',
          darknet_path + '/data/fd_02.jpg',
          darknet_path + '/data/fd_03.jpg' ]
         
thresh = 0.24
hier_thresh = 0.5

pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)

# camera 
print('----- test python API using a file')
i = 0
while i < len(files):
    filename = files[i]
    print(filename)
	# ret_val, img = cam.read()
    img = imread(filename)
    img = img.transpose(2,0,1)
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    # print w, h, c 
    data = img.ravel()/255.0
    data = np.ascontiguousarray(data, dtype=np.float32)
    outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)	
    for output in outputs:
        print(output)
    i += 1


# free model
pyyolo.cleanup()
```  

The output looks like this:
```
mask_scale: Using default '1.000000'
Loading weights from ../yolo-nfpa_1500.weights...Done!
----- test python API using a file
./darknet/data/fd_01.jpg
Cam frame predicted in 0.009999 seconds.
{'prob': 0.8797574639320374, 'bottom': 431, 'left': 740, 'right': 926, 'class': 'NFPA', 'top': 270}
./darknet/data/fd_02.jpg
Cam frame predicted in 0.009433 seconds.
{'prob': 0.8374621272087097, 'bottom': 212, 'left': 133, 'right': 302, 'class': 'NFPA', 'top': 8}
./darknet/data/fd_03.jpg
Cam frame predicted in 0.009449 seconds.
{'prob': 0.8641700148582458, 'bottom': 253, 'left': 47, 'right': 242, 'class': 'NFPA', 'top': 51}
```

Note to self: An image of size 1200x829 resulted in a segfault, though the several smaller images I used posed no problem for pyyolo ... will look into this later.  


##### Showing output images with bounding boxes  
    
Here is another version I made using skimage instead of cv2. It draws the bounding box on the image which is plotted with a simple plotting function custom_plots.py (not shown). You could just use skimage to show the images.  
```
import pyyolo
import numpy as np
import sys
from skimage.io import imread
from skimage.draw import line_aa

from sys import path as spath
spath.append('/home/davros/py_drf')
import custom_plots


darknet_path = './darknet'
datacfg = 'cfg/nfpa.data'
cfgfile = 'cfg/yolo-nfpa.2.0.cfg'
weightfile = '../yolo-nfpa_1500.weights'
files = [ darknet_path + '/data/fd_01.jpg',
          darknet_path + '/data/fd_02.jpg',
          darknet_path + '/data/fd_03.jpg' ]
         
thresh = 0.24
hier_thresh = 0.5

pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)

# camera 
print('----- test python API using a file')
i = 0
while i < len(files):
    filename = files[i]
    print(filename)
    
    dispimg = imread(filename)
    print(dispimg.shape, dispimg.dtype, dispimg[0,0,0])
    h, w, c = dispimg.shape
    img = np.rollaxis(dispimg, 2)
    print(img.shape, img.dtype, img[0,0,0])
    
    data = img.ravel()/255.0
    data = np.ascontiguousarray(data, dtype=np.float32)
    outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)	
    for output in outputs:
        print(output)
        rr, cc, val = line_aa( output['top'], output['left'], output['bottom'], output['left'] )
        dispimg[rr,cc,0] = val * 255
        dispimg[rr,cc,1] = 0
        dispimg[rr,cc,2] = 0
        
        rr, cc, val = line_aa( output['top'], output['left'], output['top'], output['right'] )
        dispimg[rr,cc,0] = val * 255
        dispimg[rr,cc,1] = 0
        dispimg[rr,cc,2] = 0
        
        rr, cc, val = line_aa( output['top'], output['right'], output['bottom'], output['right'] )
        dispimg[rr,cc,0] = val * 255
        dispimg[rr,cc,1] = 0
        dispimg[rr,cc,2] = 0
        
        rr, cc, val = line_aa( output['bottom'], output['left'], output['bottom'], output['right'] )
        dispimg[rr,cc,0] = val * 255
        dispimg[rr,cc,1] = 0
        dispimg[rr,cc,2] = 0                
        
    ans = custom_plots.show_img_return_input(dispimg, ' ')
        
    i += 1


# free model
pyyolo.cleanup()
```

An output image is shown below:  
![alt text](./images/detection.png?raw=true "Output")  


