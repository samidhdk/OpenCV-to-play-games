from Player import Player
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# import pyautogui
from threading import Thread

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

LEFT_MARGIN_THRESHOLD = int(WIDTH * (1 / 3))
RIGHT_MARGIN_THRESHOLD = int(WIDTH * (2 / 3))

CROUCH_THRESHOLD = 260
STAND_THRESHOLD = int(HEIGHT * 0.75)
JUMP_THRESHOLD = int(HEIGHT * 0.9)

# Movement Direction
MD_STAND = 0
MD_LEFT = 1
MD_RIGHT = 2


def main():
    while True:
        ret, frame = video.read()
        """If your webcam flips the image, remove the following line of code:"""
        frame = cv2.flip(frame, 1)
        results = pose.process(frame)

        # print(frame.shape)

        if results.pose_landmarks is not None:
            cv2.line(frame, (0, CROUCH_THRESHOLD), (WIDTH, CROUCH_THRESHOLD), (255, 255, 0), 2)
            cv2.line(frame, (0, STAND_THRESHOLD), (WIDTH, STAND_THRESHOLD), (255, 255, 255), 2)
            cv2.line(frame, (0, JUMP_THRESHOLD), (WIDTH, JUMP_THRESHOLD), (0, 255, 255), 2)
            # cv2.putText(frame, 'BACK', (int(WIDTH / 2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (LEFT_MARGIN_THRESHOLD, 0), (LEFT_MARGIN_THRESHOLD, HEIGHT), (0, 0, 255), 2)
            cv2.line(frame, (RIGHT_MARGIN_THRESHOLD, 0), (RIGHT_MARGIN_THRESHOLD, HEIGHT), (0, 0, 255), 2)

            landmarks = results.pose_landmarks.landmark
            player.update(landmarks)
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.circle(frame, (player.MIDDLE_CHEST, player.Y_CHEST), radius=3, color=(0, 0, 255), thickness=3)
            check_stance()
            detect_pose(frame)
            if player.cd <= 13:
                player.cd += 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def check_stance():
    if player.R_SHOULDER.y * HEIGHT > CROUCH_THRESHOLD and player.L_SHOULDER.y * HEIGHT > CROUCH_THRESHOLD:
        print("CROUCHING!")
        if player.STANCE != 1:
            player.STANCE = 1
            player.crouch()

    elif player.R_KNEE.y * HEIGHT < HEIGHT * 0.9 and player.L_KNEE.y * HEIGHT < HEIGHT * 0.9:
        print("JUMPING!")
        if player.STANCE != 2:
            player.STANCE = 2
            player.jump()
    else:
        print("NEUTRAL!")
        if player.STANCE != 0:
            player.STANCE = 0
            player.neutral()


def detect_pose(frame):
    # neutral stance

    if player.STANCE == 0:
        if player.R_ELBOW.y < player.R_SHOULDER.y and player.L_ELBOW.y < player.L_SHOULDER.y:
            print("HEAVY PUNCH!")
            player.heavy_punch()

        elif (
                int(player.L_SHOULDER.y * 0.9) < player.L_ELBOW.y < int(
            player.L_SHOULDER.y * 1.1) or player.L_ELBOW.y < player.L_SHOULDER.y) and (
                player.L_ELBOW.y < player.R_ELBOW.y):
            print("LIGHT PUNCH!")
            player.light_punch()
        elif (
                int(player.R_SHOULDER.y * 0.9) < player.R_ELBOW.y < int(
            player.R_SHOULDER.y * 1.1) or player.R_ELBOW.y < player.R_SHOULDER.y) and (
                player.R_ELBOW.y < player.L_ELBOW.y):
            print("MEDIUM PUNCH!")
            player.medium_punch()

        elif (LEFT_MARGIN_THRESHOLD < player.L_EAR.x * WIDTH < player.R_EAR.x * WIDTH < RIGHT_MARGIN_THRESHOLD
              and LEFT_MARGIN_THRESHOLD < player.MIDDLE_CHEST < RIGHT_MARGIN_THRESHOLD):
            if player.movement != MD_STAND:
                player.movement = MD_STAND
                player.stand()
                print("STAND")

        elif (player.L_EAR.x * WIDTH < player.R_EAR.x * WIDTH < LEFT_MARGIN_THRESHOLD
              and player.MIDDLE_CHEST < LEFT_MARGIN_THRESHOLD):
            if player.movement != MD_LEFT:
                player.movement = MD_LEFT
                player.move_left()
                print("MOVE LEFT")
        elif (player.R_EAR.x * WIDTH > player.L_EAR.x * WIDTH > RIGHT_MARGIN_THRESHOLD
              and player.MIDDLE_CHEST > RIGHT_MARGIN_THRESHOLD):
            if player.movement != MD_RIGHT:
                player.movement = MD_RIGHT
                player.move_right()
                print("MOVE RIGHT")


    
    # crouched stance
    elif player.STANCE == 1:
        if player.R_ELBOW.y < player.R_SHOULDER.y and player.L_ELBOW.y < player.L_SHOULDER.y:
            print("HEAVY PUNCH!")
            player.heavy_punch()

        elif (
                int(player.L_SHOULDER.y * 0.9) < player.L_ELBOW.y < int(
            player.L_SHOULDER.y * 1.1) or player.L_ELBOW.y < player.L_SHOULDER.y) and (
                player.L_ELBOW.y < player.R_ELBOW.y):
            print("LIGHT PUNCH!")
            player.light_punch()
        elif (
                int(player.R_SHOULDER.y * 0.9) < player.R_ELBOW.y < int(
            player.R_SHOULDER.y * 1.1) or player.R_ELBOW.y < player.R_SHOULDER.y) and (
                player.R_ELBOW.y < player.L_ELBOW.y):
            print("MEDIUM PUNCH!")
            player.medium_punch()
    # jumping stance
    else:
        if player.R_ELBOW.y < player.R_SHOULDER.y:
            print("HEAVY PUNCH!")
            player.heavy_punch()





if __name__ == '__main__':
    player = Player(WIDTH, HEIGHT)
    main()
