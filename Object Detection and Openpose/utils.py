# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import math
from typing import List, Tuple

import cv2
from data import Person
import numpy as np

# map edges to a RGB color
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}

# A list of distictive colors
COLOR_LIST = [
    (47, 79, 79),
    (139, 69, 19),
    (0, 128, 0),
    (0, 0, 139),
    (255, 0, 0),
    (255, 215, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (30, 144, 255),
    (255, 228, 181),
    (255, 105, 180),
]

body_part_list = ['Nose', 'Left eye', 'Right eye', 'Left ear', 'Right ear', 'Left shoulder', 'Right shoulder', 'Left elbow', 'Right elbow', 'Left wrist', 'Right wrist', 'Left hip', 'Right hip', 'Left knee', 'Right knee', 'Left ankle', 'Right ankle']

def visualize(
    image: np.ndarray,
    list_persons: List[Person],
    keypoint_color: Tuple[int, ...] = None,
    keypoint_threshold: float = 0.05,
    instance_threshold: float = 0.1,
) -> np.ndarray:
  """Draws landmarks and edges on the input image and return it.

  Args:
    image: The input RGB image.
    list_persons: The list of all "Person" entities to be visualize.
    keypoint_color: the colors in which the landmarks should be plotted.
    keypoint_threshold: minimum confidence score for a keypoint to be drawn.
    instance_threshold: minimum confidence score for a person to be drawn.

  Returns:
    Image with keypoints and edges.
  """
  print("Total person in the camera: ", len(list_persons))
  dict_track_wave_person = {}
  for person in list_persons:
  
    if person.score < instance_threshold:
      continue

    keypoints = person.keypoints
    bounding_box = person.bounding_box

    # Assign a color to visualize keypoints.
    if keypoint_color is None:
      if person.id is None:
        # If there's no person id, which means no tracker is enabled, use
        # a default color.
        person_color = (0, 255, 0)
      else:
        # If there's a person id, use different color for each person.
        person_color = COLOR_LIST[person.id % len(COLOR_LIST)]
    else:
      person_color = keypoint_color

    # Draw all the landmarks
    for i in range(len(keypoints)):
      if keypoints[i].score >= keypoint_threshold:
        cv2.circle(image, keypoints[i].coordinate, 2, person_color, 4)

    # Draw all the edges
    for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (keypoints[edge_pair[0]].score > keypoint_threshold and
          keypoints[edge_pair[1]].score > keypoint_threshold):
        
        body_part_1 = body_part_list[keypoints[edge_pair[0]][0].value]  
        #print(body_part_1, "Detected")
        body_part_1_x_coor = keypoints[edge_pair[0]].coordinate.x
        body_part_1_y_coor = keypoints[edge_pair[0]].coordinate.y
   
        #print("X Coorindate of this body part: " , str(body_part_1_x_coor))
        #print("Y Coorindate of this body part: " , body_part_1_y_coor)
        
        body_part_2 = body_part_list[keypoints[edge_pair[1]][0].value] 
        #print(body_part_2, "Detected")
        
        body_part_2_x_coor = keypoints[edge_pair[1]].coordinate.x
        body_part_2_y_coor = keypoints[edge_pair[1]].coordinate.y

        #print("X Coorindate of this body part: " , body_part_2_x_coor)
        #print("Y Coorindate of this body part: " , body_part_2_y_coor)

        # print(keypoints)
        dict_limb = {}
        dict_shoulder = {}
        limbs = ["Left wrist", "Left elbow", "Right wrist", "Right elbow"]
        shoulders = ["Left shoulder", "Right shoulder"]
        for pose in keypoints:
          if body_part_list[pose[0].value] in limbs:
            coordinate_limb_list = []
            coordinate_limb_list.append(pose[1].x)
            coordinate_limb_list.append(pose[1].y)
            dict_limb[str(body_part_list[pose[0].value])] = coordinate_limb_list
          if body_part_list[pose[0].value] in shoulders:
            coordinate_shoulder_list = []
            coordinate_shoulder_list.append(pose[1].x)
            coordinate_shoulder_list.append(pose[1].y)
            dict_shoulder[body_part_list[pose[0].value]] = coordinate_shoulder_list

        # print(dict_limb)
        # print(dict_shoulder)
        # print("-----------------------------------------")
        # print(person.id)

        for limb, limb_coordinate_list in dict_limb.items():
          for shoulder, shoulder_coordiante_list in dict_shoulder.items():
            if limb == 'Left wrist' and shoulder == 'Left shoulder':
              # print("left limb y:", limb_coordinate_list[1])
              # print("left shoulder y:", shoulder_coordiante_list[1])
              if limb_coordinate_list[1] < shoulder_coordiante_list[1]:
                print(person.id)
                print("Waving hand")
                start_x = bounding_box.start_point.x
                start_y = bounding_box.start_point.y
                end_x = bounding_box.end_point.x
                end_y = bounding_box.end_point.y
                midpoint_x = int((end_x + start_x) / 2)
                midpoint_y = int((end_y + start_y) / 2)

                dict_track_wave_person[person.id] = (midpoint_x, midpoint_y)

                cv2.circle(image, (midpoint_x,midpoint_y), 10, (0, 0, 255), 2)

            elif limb == 'Right wrist' and shoulder == 'Right shoulder':
              # print("right limb y:", limb_coordinate_list[1])
              # print("right shoulder y:", shoulder_coordiante_list[1])
              if limb_coordinate_list[1] < shoulder_coordiante_list[1]:
                print(person.id)
                print("Waving hand")
                start_x = bounding_box.start_point.x
                start_y = bounding_box.start_point.y
                end_x = bounding_box.end_point.x
                end_y = bounding_box.end_point.y
                midpoint_x = int((end_x + start_x) / 2)
                midpoint_y = int((end_y + start_y) / 2)

                dict_track_wave_person[person.id] = (midpoint_x, midpoint_y)

                cv2.circle(image, (midpoint_x,midpoint_y), 10, (0, 0, 255), 2)

        cv2.line(image, keypoints[edge_pair[0]].coordinate,
                 keypoints[edge_pair[1]].coordinate, edge_color, 2)

    # Draw bounding_box with multipose
    if bounding_box is not None:
      start_point = bounding_box.start_point
      end_point = bounding_box.end_point
      cv2.rectangle(image, start_point, end_point, person_color, 2)
      # Draw id text when tracker is enabled for MoveNet MultiPose model.
      # (id = None when using single pose model or when tracker is None)
      if person.id:
        id_text = 'id = ' + str(person.id) + " " + body_part_1 + "(" + str(body_part_1_x_coor) + "," + str(body_part_1_y_coor) + ")" + " and " + body_part_2 + "(" + str(body_part_2_x_coor) + "," + str(body_part_2_y_coor) + ")" + " Detected"
        cv2.putText(image, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 2)

  # print(dict_track_wave_person)

  return image, dict_track_wave_person

def keep_aspect_ratio_resizer(
    image: np.ndarray, target_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
  """Resizes the image.

  The function resizes the image such that its longer side matches the required
  target_size while keeping the image aspect ratio. Note that the resizes image
  is padded such that both height and width are a multiple of 32, which is
  required by the model. See
  https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1 for more
  detail.

  Args:
    image: The input RGB image as a numpy array of shape [height, width, 3].
    target_size: Desired size that the image should be resize to.

  Returns:
    image: The resized image.
    (target_height, target_width): The actual image size after resize.

  """
  height, width, _ = image.shape
  if height > width:
    scale = float(target_size / height)
    target_height = target_size
    scaled_width = math.ceil(width * scale)
    image = cv2.resize(image, (scaled_width, target_height))
    target_width = int(math.ceil(scaled_width / 32) * 32)
  else:
    scale = float(target_size / width)
    target_width = target_size
    scaled_height = math.ceil(height * scale)
    image = cv2.resize(image, (target_width, scaled_height))
    target_height = int(math.ceil(scaled_height / 32) * 32)

  padding_top, padding_left = 0, 0
  padding_bottom = target_height - image.shape[0]
  padding_right = target_width - image.shape[1]
  # add padding to image
  image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left,
                             padding_right, cv2.BORDER_CONSTANT)
  return image, (target_height, target_width)
