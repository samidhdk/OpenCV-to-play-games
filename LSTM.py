import cv2
import keras.src.distribute.saved_model_test_base
import numpy as np
import os
import mediapipe as mp
import pygetwindow as gw
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
from collections import deque
from Player import Player
import tensorflow as tf

WIDTH = 640
HEIGHT = 480

LEFT_MARGIN_THRESHOLD = int(WIDTH / 3)
RIGHT_MARGIN_THRESHOLD = int(WIDTH * (2 / 3))
CROUCH_THRESHOLD = 260
STAND_THRESHOLD = int(HEIGHT * 0.75)
JUMP_THRESHOLD = int(HEIGHT * 0.9)
# Movement Direction
MD_STAND = 0
MD_LEFT = 1
MD_RIGHT = 2
MODEL = "test-30-30-synth-data-4-movements.keras"
#MODEL = "test-30-30-synth-data-2-movements.keras"
#MODEL = "goz-1-relu-15_frames.keras"
#MODEL = "relu-15-frames-augmented.keras"
#MODEL = "relu-15-frames.keras"w

# Ruta de los logs de tensorboard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Ruta donde se guardan los datos (numpy arrays)
DATA_PATH = os.path.join('LSTM_Data')

# Acciones a detectar

actions = np.array(['fuego', 'm_punch', 'h_punch', 'idle'])


no_synth_data = 300
# 30 videos de data
no_sequences = 30
# 30 frames de información por video
waiting_frames = 60
sequences_length = 30  # 15 de video
train_frames = waiting_frames + sequences_length

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def main():
    video = cv2.VideoCapture(0)

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
    else:
        pose_array = np.zeros(33 * 4)
    return pose_array


def draw_landmarks(img, results):
    mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def process_data():
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences+no_synth_data):
            window = []
            for frame_num in range(sequences_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    x = np.array(sequences)

    y = to_categorical(labels).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test


def train_LSTM():
    x_train, _, y_train, _ = process_data()
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequences_length, 132)))  # 132 = 33* 4 (x,y,z,vis)
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))  # original relu
    model.add(Dense(32, activation='relu'))  # original relu
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=35, callbacks=tb_callback) # 105
    model.save(MODEL)

    """
    x_train, _, y_train, _ = process_data()
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(sequences_length, 132)))  # 132 = 33* 4 (x,y,z,vis)
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(32, return_sequences=False, activation='relu'))
    model.add(Dense(32, activation='relu'))  # original relu
    model.add(Dense(16, activation='relu'))  # original relu
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=50, callbacks=tb_callback) # 105
    model.save(MODEL)
    """



def confusion_matrix():
    model = keras.models.load_model(MODEL)
    x_train, x_test, y_train, y_test = process_data()

    y_test_arg = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    print('Confusion Matrix')
    print(multilabel_confusion_matrix(y_test_arg, y_pred))
    print('Accuracy Score')
    print(accuracy_score(y_test_arg, y_pred))

    return accuracy_score(y_test_arg, y_pred)


def test():
    window_title_to_focus = "Street Fighter 6 Demo"
    focus_on_application(window_title_to_focus)
    player = Player(HEIGHT=HEIGHT, WIDTH=WIDTH)
    sequence = deque(maxlen=sequences_length)
    threshold = 0.85
    video = cv2.VideoCapture(0)
    model = keras.models.load_model(MODEL)  # action.keras, relu-sigmoid.keras
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
        while True:

            ret, frame = video.read()
            frame = cv2.flip(frame, 1)
            # Make detections
            image, results = mp_detection(frame, pose_model)

            if results.pose_landmarks is not None:
                # Lines of reference
                cv2.line(image, (0, CROUCH_THRESHOLD), (WIDTH, CROUCH_THRESHOLD), (255, 255, 0), 2)
                cv2.line(image, (0, JUMP_THRESHOLD), (WIDTH, JUMP_THRESHOLD), (0, 255, 255), 2)
                cv2.line(image, (LEFT_MARGIN_THRESHOLD, 0), (LEFT_MARGIN_THRESHOLD, HEIGHT), (0, 0, 255), 2)
                cv2.line(image, (RIGHT_MARGIN_THRESHOLD, 0), (RIGHT_MARGIN_THRESHOLD, HEIGHT), (0, 0, 255), 2)

                player.update(landmarks=results.pose_landmarks.landmark)

                # Draw landmarks
                cv2.circle(image, (player.MIDDLE_CHEST_X, player.MIDDLE_CHEST_Y), radius=3, color=(0, 0, 255),
                           thickness=3)
                draw_landmarks(image, results)

                # Predict logic
                keypoints = extract_keypoints(results)
                check_stance(player)
                check_movement(player)
                if player.cd <= 13:
                    player.cd += 1

                sequence.append(keypoints)
                #print(len(sequence))

                if sum(keypoints) != 0 and player.movement == 0 and player.STANCE == 0:
                    if len(sequence) == sequences_length:

                        res = model.predict(np.expand_dims(sequence, axis=0))[0]

                        for i, action in enumerate(actions):
                            confidence = res[i] * 100
                            print(f"Acción: {action}, Confianza: {confidence:.2f}%")
                        #print(res, res[np.argmax(res)], actions[np.argmax(res)], np.argmax(res))

                        if res[np.argmax(res)] > threshold:
                            respuesta = str(actions[np.argmax(res)])
                            check_action(player, respuesta)
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


def check_stance(player):
    if player.R_SHOULDER.y * HEIGHT > CROUCH_THRESHOLD and player.L_SHOULDER.y * HEIGHT > CROUCH_THRESHOLD:

        if player.STANCE != 1:
            print("CROUCHING!")
            player.STANCE = 1
            player.crouch()

    elif player.R_KNEE.y * HEIGHT < HEIGHT * 0.9 and player.L_KNEE.y * HEIGHT < HEIGHT * 0.9:
        # elif player.MIDDLE_CHEST_Y < int(JUMP_THRESHOLD - 300):
        print("JUMPING!")
        if player.STANCE != 2:
            player.STANCE = 2
            player.jump()
    else:
        print("NEUTRAL!")
        if player.STANCE != 0:
            player.STANCE = 0
            player.neutral()


def check_movement(player):
    # neutral stance
    if player.STANCE == 0:
        if (LEFT_MARGIN_THRESHOLD < player.L_EAR.x * WIDTH < player.R_EAR.x * WIDTH < RIGHT_MARGIN_THRESHOLD
                and LEFT_MARGIN_THRESHOLD < player.MIDDLE_CHEST_X < RIGHT_MARGIN_THRESHOLD):
            if player.movement != MD_STAND:
                player.movement = MD_STAND
                player.stand()
                print("STAND")

        elif (player.L_EAR.x * WIDTH < player.R_EAR.x * WIDTH < LEFT_MARGIN_THRESHOLD
              and player.MIDDLE_CHEST_X < LEFT_MARGIN_THRESHOLD):
            if player.movement != MD_LEFT:
                player.movement = MD_LEFT
                player.move_left()
                print("MOVE LEFT")
        elif (player.R_EAR.x * WIDTH > player.L_EAR.x * WIDTH > RIGHT_MARGIN_THRESHOLD
              and player.MIDDLE_CHEST_X > RIGHT_MARGIN_THRESHOLD):
            if player.movement != MD_RIGHT:
                player.movement = MD_RIGHT
                player.move_right()
                print("MOVE RIGHT")


def check_action(player, action):
    if action == "fuego":
        player.light_punch()
    elif action == "m_punch":
        player.medium_punch()
    elif action == "h_punch":
        player.heavy_punch()


def focus_on_application(window_title):
    try:
        # Buscar la ventana por título
        window = gw.getWindowsWithTitle(window_title)

        # Verificar si se encontró la ventana
        if window:
            window = window[0]
            # Activar (poner en primer plano) la ventana
            window.activate()
            return True
        else:
            print(f"No se encontró la ventana con el título: {window_title}")
            return False
    except Exception as e:
        print(f"Error al enfocar la ventana: {e}")
        return False

def create_data_folders():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def draw_plot():
    y_plot = []
    x_plot = list(range(0, 1000))
    for i in range(0, 1000):
        y_plot.append(confusion_matrix())
    import matplotlib.pyplot as plt

    plt.plot(x_plot, y_plot)
    plt.show()

def generate_data_augmentation_pos():

    variability_range = 10
    """
    Quiero 20, 10 positivos y 10 negativos, la variabilidad va desde 0.10 - 0.20, y de -0.1 -  -0.2
    
    Cómo se va a hacer:
    
        Para cada video se va a generar 20 diferentes (variability range x2 (pos y neg))
        
        Entonces:
            - Se comienza con una acción [h_punch, idl, fuego, kick]
            - Se accede al primer video
            - Se debe seleccionar una variabilidad, se comienza por 0.1
                - Se crean 20 nuevas carpetas (20 x variabilidad(pos, neg))
                    - En la carpeta 30 se debe:
                        - coger el valor del primer video (0), ir al primer frame y sumarle esa cantidad, se guarda en 30/0
                        - se realiza para todas, obteniendo un 30/[0-29]
                    - En la carpeta 31 se debe:
                        - coger el valor del primer video (1), ir al primer frame y sumarle esa cantidad, se guarda en 30/0
            
        
        
    
    """


    #Positives
    for action in actions: # h_punch, fire, idle, tbd
        n_variability = 0.1
        dir_number = -1
        for i in range(variability_range):  # 0, 9
            n_variability += 0.01
            for number in range(no_sequences): # 0,29 Numero del video
                dir_number += 1
                os.makedirs(os.path.join(DATA_PATH, action, str(dir_number + no_sequences)))
                for frame in range(no_sequences):  # 0,29 Numero del frame
                    lpath = os.path.join(DATA_PATH, action, str(number), f"{frame}.npy")
                    print(lpath)

                    data = np.load(lpath)
                    random_values = np.full(data.shape, n_variability)

                    # Filtra los elementos en los que el valor original es distinto de cero

                    nonzero_indices = data != 0

                    # Aplica la perturbación solo en los elementos diferentes de cero

                    syntethic_data = data.copy()  # Copia los datos originales
                    syntethic_data[nonzero_indices] += random_values[nonzero_indices]
                    syntethic_data_path = os.path.join(DATA_PATH, action, str(dir_number+no_sequences), f"{frame}.npy")

                    np.save(syntethic_data_path, syntethic_data)


def generate_data_augmentation_neg():
    #Negatives
    variability_range = 10
    for action in actions: # h_punch, fire, idle, tbd
        n_variability = 0.1
        dir_number = -1
        for i in range(variability_range):  # 0, 9
            n_variability -= 0.01
            for number in range(no_sequences): # 0,29 Numero del video
                dir_number += 1
                os.makedirs(os.path.join(DATA_PATH, action, str(dir_number + no_sequences+300)))
                for frame in range(no_sequences):  # 0,29 Numero del frame
                    lpath = os.path.join(DATA_PATH, action, str(number), f"{frame}.npy")
                    print(lpath)

                    data = np.load(lpath)
                    random_values = np.full(data.shape, n_variability)

                    # Filtra los elementos en los que el valor original es distinto de cero

                    nonzero_indices = data != 0

                    # Aplica la perturbación solo en los elementos diferentes de cero

                    syntethic_data = data.copy()  # Copia los datos originales
                    syntethic_data[nonzero_indices] += random_values[nonzero_indices]
                    syntethic_data_path = os.path.join(DATA_PATH, action, str(dir_number+no_sequences+no_synth_data), f"{frame}.npy")

                    np.save(syntethic_data_path, syntethic_data)


if __name__ == '__main__':
    #create_data_folders() # Crea las carpetas para guardar los datos.
    #main() # Graba las acciones y se guardan en sus correspondientes carpetas.

    #train_LSTM() #Entrena la red.
    #confusion_matrix()

    #generate_data_augmentation_pos();generate_data_augmentation_neg()

    test()  # prueba la red con los datos obtenidos.
