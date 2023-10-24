from Player import Player
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
"""
# Ruta donde se guardan los datos (numpy arrays)
DATA_PATH = os.path.join('LSTM_Data')
#Acciones a detectar
actions = np.array(['l_punch', 'm_punch', 'h_punch'])

# 30 videos de data
no_sequences = 30
# 30 frames de información por video
sequences_length = 30




mp_pose = mp.solutions.pose
pose = mp_pose.Pose()



def main():
    video = cv2.VideoCapture(0)


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_number in range(sequences_length):
                    ret, frame = video.read()
                    frame = cv2.flip(frame, 1)

                    #Make detections
                    img, results = mp_detection(frame, pose_model)

                    #Draw landmarks
                    draw_landmarks(img, results)

                    if frame_number == 0:
                        cv2.putText(frame, 'PREPARE', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(frame, 'Collection frames for {} Video Nº{}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 4, cv2.LINE_AA)

                        #cv2.waitKey(2000)

                    else:
                        cv2.putText(frame, 'Collection frames for {} Video Nº{}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 4, cv2.LINE_AA)

                    cv2.imshow('LSTM', img)
                    """
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame))
                    np.save(npy_path, keypoints)
                    """
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    continue
                break
            else:
                continue
            break

        video.release()
        cv2.destroyAllWindows()


def mp_detection(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, results

def extract_keypoints(results):
    pose_array = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    return pose_array
def draw_landmarks(img, results):
    mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    """
    mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    """
def create_data_folders():
    for action in actions:
        for sequence in range(sequences_length):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass


if __name__ == '__main__':
    create_data_folders()
    main()
