import pydirectinput


class Player:
    def __init__(self, WIDTH, HEIGHT):
        self.R_EAR = None
        self.L_EAR = None
        self.R_ELBOW = None
        self.L_ELBOW = None
        self.R_SHOULDER = None
        self.L_SHOULDER = None
        self.R_HAND = None
        self.L_HAND = None
        self.R_WAIST = None
        self.L_WAIST = None
        self.R_KNEE = None
        self.L_KNEE = None
        self.STANCE = 0
        self.MIDDLE_CHEST = None
        self.Y_CHEST = None
        self.cd = 13
        self.movement = 0  # 0 stand, 1 left, 2 right
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
    def update(self, landmarks):
        self.R_EAR = landmarks[7]
        self.L_EAR = landmarks[8]
        self.R_ELBOW = landmarks[13]
        self.L_ELBOW = landmarks[14]
        self.R_SHOULDER = landmarks[11]
        self.L_SHOULDER = landmarks[12]
        self.R_HAND = landmarks[15]
        self.L_HAND = landmarks[16]
        self.R_WAIST = landmarks[23]
        self.L_WAIST = landmarks[24]
        self.R_KNEE = landmarks[25]
        self.L_KNEE = landmarks[26]
        self.MIDDLE_CHEST = int((self.L_SHOULDER.x + self.R_SHOULDER.x) * self.WIDTH / 2)
        self.Y_CHEST = int((self.L_SHOULDER.y + self.R_SHOULDER.y) * self.HEIGHT / 2)

    def light_kick(self):
        if self.cd >= 5:
            pydirectinput.press('j')
            self.cd = 0

    def medium_kick(self):
        if self.cd >= 8:
            pydirectinput.press('k')
            self.cd = 0

    def heavy_kick(self):
        if self.cd >= 13:
            pydirectinput.press('l')
            self.cd = 0

    def light_punch(self):

        if self.cd >= 5:
            pydirectinput.press('u')
            self.cd = 0

    def medium_punch(self):
        if self.cd >= 8:
            pydirectinput.press('i')
            self.cd = 0

    def heavy_punch(self):
        if self.cd >= 13:
            pydirectinput.press('o')
            self.cd = 0

    def crouch(self):

        pydirectinput.keyDown('s')

    def jump(self):

        pydirectinput.keyUp('s')
        pydirectinput.press('w')

    def neutral(self):
        pydirectinput.keyUp('s')

    def stand(self):  # 0
        pydirectinput.keyUp('s')
        pydirectinput.keyUp('a')  # :(
        pydirectinput.keyUp('d')

    def move_left(self):  # 1
        pydirectinput.keyUp('d')
        pydirectinput.keyDown('a')

    def move_right(self):  # 2
        pydirectinput.keyUp('a')
        pydirectinput.keyDown('d')
