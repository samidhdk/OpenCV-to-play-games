import time

import cv2
import keras.src.distribute.saved_model_test_base
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
from collections import deque
import tensorflow as tf

MODEL = "relu-15-frames.keras"

# Ruta de los logs de tensorboard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Ruta donde se guardan los datos (numpy arrays)
DATA_PATH = os.path.join('LSTM_Data_15')

# Acciones a detectar
actions = np.array(['l_punch', 'm_punch', 'h_punch', 'idle'])
# actions = np.array(['idle'])

# 30 videos de data
no_sequences = 30
# 30 frames de información por video
waiting_frames = 60
sequences_length = 15  # 15 de video
train_frames = waiting_frames + sequences_length

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def main():
    video = cv2.VideoCapture(1)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
        for action in actions:
            for sequence in range(no_sequences):

                for frame_number in range(train_frames):

                    ret, frame = video.read()
                    frame = cv2.flip(frame, 1)

                    # Make detections
                    img, results = mp_detection(frame, pose_model)

                    # Draw landmarks
                    draw_landmarks(img, results)

                    if frame_number < waiting_frames:
                        cv2.putText(img, f'PREPARE {sequence + 1} / {no_sequences}', (70, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(img, 'Collection frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('LSTM', img)

                    else:
                        cv2.putText(img, 'Collection frames for {} Video Number {}'.format(action, sequence + 1),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        cv2.imshow('LSTM', img)

                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_number - waiting_frames))
                        print(frame_number - waiting_frames)
                        np.save(npy_path, keypoints)

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
    if results.pose_landmarks:
        pose_array = np.array(
            [[res.x, res.y, res.z, res.visibility] if res.visibility > 0.9 else [0, 0, 0, 0] for res in
             results.pose_landmarks.landmark]).flatten()
        # vis_arr = np.array(
        #    [[res.visibility] if res.visibility > 0.9 else [0] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose_array = np.zeros(33 * 4)
        # vis_arr = np.zeros(33 * 4)

    # a = sum(1 for elem in vis_arr if elem > 0)
    # print(a)
    return pose_array


def draw_landmarks(img, results):
    mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def process_data():
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequences_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    x = np.array(sequences)

    y = to_categorical(labels).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    print(np.shape(x_train))
    print("##############")
    print(np.shape(x_test))
    print("##############")
    print(np.shape(y_train))
    print("##############")
    print(np.shape(y_test))
    return x_train, x_test, y_train, y_test


def train_LSTM():
    x_train, x_test, y_train, y_test = process_data()
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(15, 132)))  # 132 = 33* 4 (x,y,z,vis)
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(LSTM(256, return_sequences=False, activation='relu'))
    model.add(Dense(128, activation='relu'))  # original relu
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))  # original relu
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=600, callbacks=tb_callback)
    model.save(MODEL)


def create_data_folders():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass


def test():
    sequence = deque(maxlen=sequences_length)
    threshold = 0.95
    video = cv2.VideoCapture(1)
    model = keras.models.load_model(MODEL)  # action.keras, relu-sigmoid.keras
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
        while True:

            ret, frame = video.read()
            frame = cv2.flip(frame, 1)

            # Make detections
            image, results = mp_detection(frame, pose_model)

            # Draw landmarks
            draw_landmarks(image, results)

            # Predict logic
            keypoints = extract_keypoints(results)
            if sum(keypoints) != 0:
                sequence.append(keypoints)
                print(len(sequence))

                if len(sequence) == sequences_length:

                    res = model.predict(np.expand_dims(sequence, axis=0))[0]

                    for i, action in enumerate(actions):
                        confidence = res[i] * 100
                        print(f"Acción: {action}, Confianza: {confidence:.2f}%")
                    # print(res, res[np.argmax(res)], actions[np.argmax(res)], np.argmax(res))

                    if res[np.argmax(res)] > threshold:
                        respuesta = str(actions[np.argmax(res)])
                        media.append(np.argmax(res))
                        print(np.argmax(res), media)
                    else:
                        respuesta = ""

                    cv2.rectangle(image, (0, 0), (320, 40), (245, 117, 16), -1)
                    cv2.putText(image, respuesta, (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('LSTM', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # create_data_folders()
    # main()
    # train_LSTM()
    test()
