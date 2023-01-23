#!/usr/bin/env python3
import numpy as np
import cv2
import time
import logging
import sys
from ml import MoveNetMultiPose
import utils

class Unity_Simulation():

    def __init__(self):
        cv2.namedWindow("arena_simulation", 1)
        self.font_scale = 3
        self.font = cv2.FONT_HERSHEY_PLAIN

        # OIV4 model
        self.config_file_601 = './models/graph_601.pbtxt'
        self.frozen_model_601 = './models/frozen_inference_graph_601.pb'
        self.file_name_601 = './models/objects.names_601.en'
        self.model_object_detection_601 = cv2.dnn.readNetFromTensorflow(self.frozen_model_601, self.config_file_601)
        self.classLabels_601 = []
        with open(self.file_name_601, 'rt') as fpt:
            self.classLabels_601 = fpt.read().rstrip('\n').split('\n')
        self.colors_oidv4 = np.random.uniform(0, 255, size=(len(self.classLabels_601), 3))

        # COCO 2017 model
        self.classes_coco_2017 = []
        self.frozen_model_coco_2017  = './models/frozen_inference_graph_coco_2017.pb'
        self.config_file__coco_2017 = './models/ssd_inception_v2_coco_2017_11_17.pbtxt'
        self.file_name_coco_2017 = './models/objects_coco_2017.txt'
        self.model_object_detection_coco_2017 = cv2.dnn.readNetFromTensorflow(self.frozen_model_coco_2017,self.config_file__coco_2017)   
        with open(self.file_name_coco_2017, 'rt') as fpt:
            self.classes_coco_2017 = fpt.read().rstrip('\n').split('\n')
        self.colors_coco = np.random.uniform(0, 255, size=(len(self.classes_coco_2017), 3))


    def run(self,estimation_model: str, tracker_type: str, width: int, height: int, frame, image) -> None:
	
        # Notify users that tracker is only enabled for MoveNet MultiPose model.
        if tracker_type and (estimation_model != 'movenet_multipose'):
            logging.warning(
            'No tracker will be used as tracker can only be enabled for '
            'MoveNet MultiPose model.')

        # Initialize the pose estimator selected.
        estimation_model == 'movenet_multipose'
        pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
        
        list_persons = pose_detector.detect(image)

        # Draw keypoints and edges , body joints, x , y coordinates on input image
        image = utils.visualize(image, list_persons)

        return image

    def image_callback(self,image):
        
        height = image.shape[0]
        width = image.shape[1]

        ############# Object Detection #############
        ############# OIV4 #############
        blob_oiv4 = cv2.dnn.blobFromImage(image, size=(
            300, 300), swapRB=True, crop=False)
        self.model_object_detection_601.setInput(blob_oiv4)
        outputs_oiv4 = self.model_object_detection_601.forward()
        for detection in outputs_oiv4[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.3:
                classID = int(detection[1])
                classe = self.classLabels_601[classID - 1]
                left = int(detection[3] * width)
                top = int(detection[4] * height)
                right = int(detection[5] * width)
                bottom = int(detection[6] * height)
                print(classe + " Detected")
                cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
                cv2.putText(image, str(classe), (int(left), int(top)), self.font, self.font_scale, self.colors_oidv4[classID], 2)

        ############# COCO 2017 #############

        blob_coco = cv2.dnn.blobFromImage(image, size=(
            300, 300), swapRB=True, crop=False)
        self.model_object_detection_coco_2017.setInput(blob_coco)
        cvOut = self.model_object_detection_coco_2017.forward()

        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.3:
                idx = int(detection[1])   
                left = detection[3] * width
                top = detection[4] * height
                right = detection[5] * width
                bottom = detection[6] * height
                cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                label = "{}: {:.2f}%".format(self.classes_coco_2017[idx],score * 100)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, label, (int(left), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors_coco[idx], 2)
                print(self.classes_coco_2017[idx] + " Detected")

        frame = np.array(image, dtype=np.uint8)
        frame = cv2.resize(frame, (640, 480))
        model = 'movenet_multipose'
        tracker = 'bounding_box'
        frameWidth = 640
        frameHeight = 480
        image = self.run(model, tracker, frameWidth, frameHeight, frame, image)

        return image


def main(args=None):

    cap = cv2.VideoCapture('/home/crossing/Desktop/Louis_git/AI_Projects/Object Detection and Openpose/example video/streetview.gif')
    while cap.isOpened():
        ret, img = cap.read()
        simulation = Unity_Simulation()
        image = simulation.image_callback(img)
        cv2.imshow("arena_simulation", image[0])            
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

