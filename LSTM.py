import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split




# Ruta donde se guardan los datos (numpy arrays)
DATA_PATH = os.path.join('LSTM_Data')
# Acciones a detectar
actions = np.array(['l_punch', 'm_punch', 'h_punch'])

# 30 videos de data
no_sequences = 30
# 30 frames de información por video
sequences_length = 90  # 60 de espera y 30 de video, por lo que se debe restar 60

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

                    # Make detections
                    img, results = mp_detection(frame, pose_model)

                    # Draw landmarks
                    draw_landmarks(img, results)

                    if frame_number < 60:
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
                        """
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_number-60))
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
    pose_array = np.array([[res.x, res.y, res.z, res.visibility] for res in
                           results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    return pose_array


def draw_landmarks(img, results):
    mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def process_data():
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequences_length - 60):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    x = np.array(sequences)
    x = x.shape
    y = to_categorical(labels).astype(int)

    print(x, y)



def create_data_folders():
    for action in actions:
        for sequence in range(sequences_length - 60):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass


if __name__ == '__main__':
    process_data()
    #create_data_folders()
    #main()
