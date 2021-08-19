import pyautogui
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
    SIZE_X = 10
    SIZE_Y = 22
    debug = False
    graphic = False
    liner = True
    control = True
    antiBan = False
    soloGame = {'left': 653, 'top': 200 - 67 , 'width': 706, 'height': 746 + 67}
    fildLeft = 175
    fildTop = 80
    fildRight = 511
    fildBottom = 746+67
    fildWeight = fildRight - fildLeft
    fildHeight = fildBottom - fildTop
    low = np.array([0,0,110])
    high = np.array([255,255,255])
    fildWeight = 511 - 175
    fildHeight = 746 - 80 + 67

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
    kh = 4
    kd =  200
    kLines = 300
    kCount = 100000
    sum = 0

    closer = [0,0,0,0,0,0,0,0,0,0]
    for y in range(0, setting.SIZE_Y):
        for x in range(0, setting.SIZE_X):
            if(fild[y][x] == 1):
                closer[x] = 1
                sum += (y - 10) * kh

            if (fild[y][x] == 0 and closer[x] == 1):
                sum -= kd * (setting.SIZE_Y - y + 1)
    sum += fildAndLines[1] * kLines
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

