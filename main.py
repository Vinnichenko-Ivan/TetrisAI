import pyautogui
import configparser
import tkinter
import time
from enum import Enum
import numpy as np
import cv2
from mss import mss
from PIL import Image
import copy
from random import uniform
from random import randint




class Setting():
    askScreenPos = False
    SIZE_X = 10
    SIZE_Y = 22
    debug = False
    graphic = False
    liner = False
    control = True
    antiBan = False
    soloGame = {'left': 0, 'top': 0 , 'width': 0, 'height': 0}
    fildLeft = 0
    fildTop = 0
    fildRight = 0
    fildBottom = 0
    fildWeight = fildRight - fildLeft
    fildHeight = fildBottom - fildTop
    low = np.array([0,0,110])
    high = np.array([255,255,255])
    fildWeight = 511 - 175
    fildHeight = 746 - 80 + 67

    mouseXStart = 0
    mouseYStart = 0
    mouseXEnd = 0
    mouseYEnd = 0
    mouseX = 0
    mouseY = 0
    rectGetted = False

    def mouseCallBack(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseXStart = x
            self.mouseYStart = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouseXEnd = x
            self.mouseYEnd = y
            self.rectGetted = True
        elif event == cv2.EVENT_MOUSEMOVE:
            self.mouseX = x
            self.mouseY = y
        print('mouse')

    def __init__(self):
        config = configparser.ConfigParser()
        config.sections()
        config.read('conf.ini')
        self.askScreenPos = "getWindowSize" in config["SCREEN"]
        if("UseConfWindowSize" in config["SCREEN"]):
            self.soloGame['left'] = int(config['SCREEN']['left'])
            self.soloGame['top'] = int(config['SCREEN']['top'])
            self.soloGame['width'] = int(config['SCREEN']['width'])
            self.soloGame['height'] = int(config['SCREEN']['height'])
            self.fildLeft = int(config['SCREEN']['fildLeft'])
            self.fildTop = int(config['SCREEN']['fildTop'])
            self.fildRight = int(config['SCREEN']['fildRight'])
            self.fildBottom = int(config['SCREEN']['fildBottom'])
            self.fildWeight = self.fildRight - self.fildLeft
            self.fildHeight = self.fildBottom - self.fildTop
        self.debug = "debug" in config["DEBUG"]
        self.graphic = "graphic" in config["DEBUG"]
        self.liner = "liner" in config["DEBUG"]



        if(self.askScreenPos):
            root = tkinter.Tk()
            root.withdraw()
            WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
            print(WIDTH, HEIGHT)
            allScreen = {'left': 0, 'top': 0, 'width': WIDTH, 'height': HEIGHT}
            screenShot = mss().grab(allScreen)
            img = np.array(Image.frombytes('RGB', (screenShot.width, screenShot.height), screenShot.rgb, ))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            cv2.imshow('test', img)
            cv2.setMouseCallback('test', self.mouseCallBack)
            while(not self.rectGetted):
                cv2.imshow('test', img)
                if cv2.waitKey(33) & 0xFF in (ord('q'), 27,):
                     break

            self.soloGame['left'] = self.mouseXStart
            self.soloGame['top'] = self.mouseYStart
            self.soloGame['width'] = self.mouseXEnd - self.mouseXStart
            self.soloGame['height'] = self.mouseYEnd - self.mouseYStart
            self.mouseXStart = 0
            self.mouseYStart = 0

            self.rectGetted = False
            while(not self.rectGetted):
                screenShot = mss().grab(self.soloGame)
                img = np.array(Image.frombytes('RGB', (screenShot.width, screenShot.height), screenShot.rgb, ))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                cv2.rectangle(img, (self.mouseXStart, self.mouseYStart), (self.mouseX, self.mouseY), (0, 255, 0), 5)
                cv2.imshow('test', img)

                if cv2.waitKey(33) & 0xFF in (ord('q'), 27,):
                    break

            self.fildLeft = self.mouseXStart
            self.fildTop = self.mouseYStart
            self.fildRight = self.mouseXEnd
            self.fildBottom = self.mouseYEnd
            self.fildWeight = self.fildRight - self.fildLeft
            self.fildHeight = self.fildBottom - self.fildTop
            cv2.destroyWindow("test")



setting = Setting()

class Tetramino(Enum):
    I = 0
    J = 1
    L = 2
    O = 3
    S = 4
    T = 5
    Z = 6
    No = 7

def getMatrix(type):
    if(type == Tetramino.I):
        answer = [[0, 1, 0,0],[0,1, 0,0],[0,1 ,0,0],[0,1, 0,0]]
    elif(type == Tetramino.J):
        answer = [[1, 1, 0],[0, 1, 0],[0 ,1, 0]]
    elif(type == Tetramino.L):
        answer = [[0, 1, 0],[0, 1, 0],[1 ,1, 0]]
    elif(type == Tetramino. O):
        answer = [[1, 1],[1, 1]]
    elif(type == Tetramino.Z):
        answer = [[0, 1, 0],[1, 1, 0],[1 ,0, 0]]
    elif(type == Tetramino.T):
        answer = [[0, 1, 0],[1, 1, 0],[0 ,1, 0 ]]
    elif(type == Tetramino.S):
        answer = [[1, 0, 0],[1, 1, 0],[0, 1, 0]]
    return answer

def newTetramino(fild):
    for h in range(0, 4):
        if(np.sum(fild[h]) + np.sum(fild[h + 1]) == 4):
            for i in range(0, 6):
                if(fild[h][i] == 1 and fild[h][i + 1] == 1 and fild[h][i + 2] == 1 and fild[h][i + 3] == 1):
                    return [Tetramino.I, i]

            for i in range(0, 7):
                if(fild[h + 1][i] == 1 and fild[h + 1][i + 1] == 1 and fild[h + 1][i + 2] == 1 and fild[h][i] == 1):
                    return [Tetramino.J, i]

                elif(fild[h + 1][i] == 1 and fild[h + 1][i + 1] == 1 and fild[h + 1][i + 2] == 1 and fild[h][i + 2] == 1):
                    return [Tetramino.L, i]

                elif(fild[h + 1][i] == 1 and fild[h + 1][i + 1] == 1 and fild[h][i + 1] == 1 and fild[h][i + 2] == 1):
                    return [Tetramino.Z, i ]

                elif(fild[h + 1][i] == 1 and fild[h + 1][i + 1] == 1 and fild[h + 1][i + 2] == 1 and fild[h][i + 1] == 1):
                    return [Tetramino.T, i]

                elif(fild[h][i] == 1 and fild[h][i + 1] == 1 and fild[h + 1][i + 1] == 1 and fild[h + 1][i + 2] == 1):
                    return [Tetramino.S, i ]

            for i in range(0, 8):
                if (fild[h + 1][i] == 1 and fild[h + 1][i + 1] == 1 and fild[h][i] == 1 and fild[h][i + 1] == 1):
                    return [Tetramino.O, i]

    return [Tetramino.No, 0]

def copyMatrixOnFild(matrix, fild, i, h):
    for xC in range(0, len(matrix)):
        for yC in range(0, len(matrix[0])):
            if (matrix[xC][yC] == 1):
                fild[h + yC][i + xC] = 1;
                if (i + xC < 0):
                    return None
    return fild



def addNewTetramino(fild, i, a, newFigure):
    try:
        matrix = np.rot90(getMatrix(newFigure), -a)
        buff = fild
        if(i == -1 and any(matrix[0][temp] == 1 for temp in len((matrix[0])))):
            return None

        for h in range(0, setting.SIZE_Y):
            for x in range(0, len(matrix)):
                for y in range(0, len((matrix[0]))):
                    if(matrix[x][y] == 1):
                        if(h + y + 1 == setting.SIZE_Y):
                            return copyMatrixOnFild(matrix, buff, i, h)
                        if (buff[h + y + 1][i + x] == 1):
                            return copyMatrixOnFild(matrix, buff, i, h)
    except:
        return None

def testLineDelete(fild):
    lineSum = 0
    for h in range(setting.SIZE_Y - 1, -1, -1):
        if(all(fild[h][x] == 1 for x in range(0, setting.SIZE_X))):
            lineSum += 1
            for hc in range(h - 1, -1, -1):
                for x in range(0, setting.SIZE_X):
                    fild[hc + 1][x] = 0
                    fild[hc + 1][x] = fild[hc][x]
    return [fild, lineSum]


def genAllFilds(fild, newFigure):
    answer = []
    print(newFigure)
    for i in range(-3 , setting.SIZE_X + 2):
        for a in range(0, 4):
            add = copy.deepcopy(fild)
            buff = addNewTetramino(add,  i , a, newFigure)
            if(buff != None):
                answer.append([i, a, buff]);
    return answer


def areaWhitePercent(img):
    count = 0
    count = np.sum(img)
    return count / len(img) / len(img[0]) / 255 * 100

def q(fildAndLines):
    fild = fildAndLines[0]
    kh = 5
    kd =  10000000
    kMaxHeight = 100
    kLines = 1000000
    kCount = 1
    sum = 0
    kPit = 10000

    closer = []
    for x in range(0, setting.SIZE_X):
        closer.append(0)

    for y in range(0, setting.SIZE_Y):
        for x in range(0, setting.SIZE_X):
            if(fild[y][x] == 1):
                closer[x] += 1
                sum -= (setting.SIZE_Y - y) * kh

            if (fild[y][x] == 0 and closer[x] != 0):
                sum -= kd

    high = []
    for x in range(0, setting.SIZE_X):
        high.append(0)
    for x in range(0, setting.SIZE_X):
        for y in range(0, setting.SIZE_Y):
            if(fild[y][x] == 1):
                high[x] = (setting.SIZE_Y - y)
                break
    sum -= kMaxHeight*max(high)
    countPit = 0
    pitHigh = 1
    pitMass = []
    for x in range(0, setting.SIZE_X):
        pitMass.append(0)
    for x in range(0, setting.SIZE_X):
        if(x == 0):
            if(high[1] - high[0] > pitHigh):
                pitMass[x] = high[1] - high[0]
        elif (x == setting.SIZE_X - 1):
            if (high[setting.SIZE_X - 2] - high[setting.SIZE_X - 1] > pitHigh):
                # pitMass[x] = high[setting.SIZE_X - 2] - high[setting.SIZE_X - 1]
                a = 0
        elif(max(high[x - 1] - high[x], high[x + 1] - high[x]) > pitHigh):
            pitMass[x] = max(high[x - 1] - high[x], high[x + 1] - high[x])

    for i in pitMass:
        sum -= max((i - pitHigh), 0) * kPit

    for y in range(0, setting.SIZE_Y):
        if(np.sum(fild[y]) != 0):
            sum -= kMaxHeight * y
            break



    sum += ((fildAndLines[1]) * kLines)
    sum -= np.sum(np.sum(fild)) * kCount
    return sum


def nothing(x):
 pass

if(setting.debug):
    cv2.namedWindow('image')
    cv2.createTrackbar('H_H','image',0,255,nothing)
    cv2.createTrackbar('H_L','image',0,255,nothing)
    cv2.createTrackbar('S_H','image',0,255,nothing)
    cv2.createTrackbar('S_L','image',0,255,nothing)
    cv2.createTrackbar('V_H','image',0,255,nothing)
    cv2.createTrackbar('V_L','image',0,255,nothing)

def getScreenShotHSV():
    screenShot = mss().grab(setting.soloGame)
    img = np.array(Image.frombytes('RGB', (screenShot.width, screenShot.height), screenShot.rgb, ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


def openCvGetFild():
    img = getScreenShotHSV()
    gameFild = img[setting.fildTop:setting.fildBottom, setting.fildLeft:setting.fildRight]
    #hold = img[106:207, 4:175]
    #next = img[0:746, 175:511]
    if (setting.debug):
        setting.high[0] = cv2.getTrackbarPos('H_H', 'image')
        setting.high[1] = cv2.getTrackbarPos('S_H', 'image')
        setting.high[2] = cv2.getTrackbarPos('V_H', 'image')
        setting.low[0] = cv2.getTrackbarPos('H_L', 'image')
        setting.low[1] = cv2.getTrackbarPos('S_L', 'image')
        setting.low[2] = cv2.getTrackbarPos('V_L', 'image')

    gameFildBin = cv2.inRange(gameFild, setting.low, setting.high)
    #holdBin = cv2.inRange(hold, low, high)
    #nextBin = cv2.inRange(next, low, high)

    fildList = []
    for i in range(0, setting.SIZE_Y):
        fildList.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for y in range(0, setting.SIZE_Y):
        for x in range(0, setting.SIZE_X):
            a = 0
            if (areaWhitePercent(gameFildBin[y * setting.fildHeight // setting.SIZE_Y:(y + 1) * setting.fildHeight // setting.SIZE_Y,
                                 x * setting.fildWeight // setting.SIZE_X:(x + 1) * setting.fildWeight // setting.SIZE_X]) > 50):
                fildList[y][x] = 1

    if (setting.graphic):
        if (setting.liner):
            for i in range(0, setting.fildWeight, setting.fildWeight // setting.SIZE_X):
                cv2.line(gameFildBin, (i, 0), (i, setting.fildHeight), (255), thickness=1)
            for i in range(0, setting.fildHeight, setting.fildHeight // setting.SIZE_Y):
                cv2.line(gameFildBin, (0, i), (setting.fildWeight, i), (255), thickness=1)
        #cv2.imshow('test', img)
        cv2.imshow('filg', gameFild)
        #cv2.imshow('hold', hold)
        #cv2.imshow('next', next)
        cv2.imshow('gameFildBin', gameFildBin)
        #cv2.imshow('holdBin', holdBin)
        #cv2.imshow('nextBin', nextBin)
    return fildList

def getOptimalPosithion(pothithions):
    optimalPothithion = pothithions[0]
    maximum = -99999999999999
    for i in pothithions:
        answ = q(testLineDelete(i[2]))
        if (answ > maximum):
            maximum = answ
            optimalPothithion = i
    return optimalPothithion

while True :
    fildList = openCvGetFild()
    newFigure = newTetramino(fildList)

    for i in range(0, setting.SIZE_X):
        fildList[0][i] = 0
        fildList[1][i] = 0
        fildList[2][i] = 0
        fildList[3][i] = 0
        fildList[4][i] = 0

    if(newFigure[0] != Tetramino.No):
        optimal = getOptimalPosithion(genAllFilds(fildList.copy(), newFigure[0]))

        if(setting.control):

            time.sleep(0.01)
            for i in range(optimal[1]):
                pyautogui.press('z')
                if (setting.antiBan):
                    time.sleep(uniform(0.1, 0.3))

            if(optimal[0] - newFigure[1] > 0):
                for i in range(optimal[0] - newFigure[1]):
                    pyautogui.press('right')
                    if (setting.antiBan):
                        time.sleep(uniform(0.1, 0.3))

            if (optimal[0] - newFigure[1] < 0):
                for i in range(newFigure[1] - optimal[0]):
                    pyautogui.press('left')
                    if(setting.antiBan):
                        time.sleep(uniform(0.1, 0.3 ))


            pyautogui.press(' ')
            if(setting.antiBan):
                time.sleep(uniform(0.4, 0.5))

    if cv2.waitKey(33) & 0xFF in (ord('q'),27,):
        break

