# Tensorflow Object Detection / Openpose

This respository provides general computer vision algorithms for object detection, pose detection and human/object tracking.  

## Tensorflow Object Detection 
### General Description: 
The ssd_mobilenet_v2 oid v4 model is a pretrained model based on OpenImage version 4. It is capable of detecting 601 classes. Supporting objects are shown in objects.names.txt. This tensorflow object detection API, created by Google, is an opensource framework which is very useful to the computer vision project. 

The ssd_inception_v2_coco_2017 model is a pretrained model based on CoCo dataset 2017. This model is capable of detecting 80 classes. However, this model detect human, kitchen utensils or other small predefined small objects better than the ssd_mobilenet_v2 oid v4 model.

Combination of two models running simultaneously together will optimise the object detection capability. Threshold can be set differently to control false positive and false negative. 

## Tensorflow / Tensorflow Lite Openpose - detecting human joints and tracking human
### General Description: 
This project implements Movenet pose estimation model, which is the latest pose estimation model from tensorflow, to detect keypoints of a human based on the confidence score from 0 to 1. The score indicates the probability of detecting a keypoint in a certain position, such as hip or ankle. There are two versions of multipose algorithms available at this repository, one is tensorflow version (Object_detection_openpose.py) and the other one is tensorflow lite version (movenet folder). 

### Tensorflow Lite Openpose - human track
The tensorflow lite version is capable of tracking human and detecting human joints at the same time. 

# Object Tracker
### General Description: 
This project implements OpenCV object detector API including 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW' and 'CSRT'. Each detector has its own strength and weakness. My algorithm here is to use multiple detectors together (based on your preference) to take their strenths at one go. MobileNetSSD caffemodel is used here to detect object.  

### Demo of object detection and openpose (ssd_mobilenet_v2 oid v4 and tensorflow openpose)
<p align="center"><img src="ironman.gif"\></p>

# Customised Object Detection Model - Yolov3/Yolov4
### Steps for training a customised Yolov3 tiny object detection model (same steps for Yolov4 as well)
#### Use computer webcam to collect image data 

1. Open capture_webcam_images.py in the yolo_label folder
2. Change path on line 21 for directory that we want to save our images
3. capture_frame = 10 means capture one image every 10 frames

#### Label image data (Annotation)

1. Open yolo_label folder and go to labelimg/data .
2. ***Modify predefined_classes.txt for the detecting classes*** Make sure modify it, otherwise, wrong class index. Class index order does not matter.
3. Go back to the main labelimg folder and Run labelimg.py in the terminal 
4. Change to YOLO compatible annotation format on the column
5. Press Dir where the original images locate from previous step
6. Press Change Save Dir to set where you want to save the image annotation files
7. Start manually draw bounding box on the image (You can do multiclass annotation on one image)
8. After finish drawing bounding box, you should see text files located in the folder set in step 6.
9. Merge all images and annotation files into one folder. 
10. Compress all images and annotation files into a zip file, name the zip file as images.zip.

*****An example of Apple, Lemon, all_images_text folders are located in the yolo_label folder for your reference. 
The all_images_text folder is the final prepared product for images and annotation prior to training. 
You can see there is a images.zip file inside the all_images_text folder which is ready to be uploaded on Google drive for training in Google Colab in then next step. Once you finished labelling and merging all image and text files, you should have a similar zip file *****

#### Training

Open yolov3_tiny_custom.ipynb file on Google Colab
Go to Edit, then go to Notebook setting, make sure GPU is selected.

1. Create a folder on google drive and named it as yolov3 and upload the images.zip file to this yolov3 folder
2. Run yolov3_tiny_custom.ipynb on Google Colab
3. Run cell one by one until the training cell (Don't train yet)
4. There are some bugs in this ipynb file for modifying num of classes in obj.data. So, we need to modify it ourselves by opening the obj.data file located in yolotinyv3_medmask_demo
4. Modify the 'backup' directory in obj.data (located in yolotinyv3_medmask_demo) and replace 'backup/' to your google drive directory (So,that every updated weight will be saved to your google drive directly)
5. Make sure the images and text files are unzipped and copied to /content/yolotinyv3_medmask_demo/obj
6. Make sure the train.txt and valid.txt are made. You should see something like this as follow for example. 
   
   Your image file extension is: .jpg
   Number of images: 6846
   Number of images used for training 6162
   Number of images used for validation 342
   Number of images used for testing 342
  
7. Make sure the content in the obj.data and obj.names are all correct.

***** (Preferred) Training on Google Colab 
(make sure to use subscripted account, otherwise, the runtime will be shutdown in 6 hours)

1. Start training (run the training cell)
2. To test the weight file, run the yolo_object_detection.py (change all the path for weights,cfg file and label file)

#### Training on own computer

Follow instruction here: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

1. Make sure we have darknet directory ready in the computer
2. Unzip the image and text files in the obj folder of darknet directory
3. Create train, validation txt file (Either follow instruction from the ipynb file or follow the instruction from AlexeyAB)
4. Modify obj.data (where classes = number of objects, backup is where you want to save the weight file)
5. Modify obj.names (all class names)
6. Download yolov3-tiny.conv.15
7. Modify yolov3-tiny_obj.cfg  (instruction from the ipynb file)
8. Start training  
