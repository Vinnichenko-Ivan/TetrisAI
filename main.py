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

SIZE_X = 10
SIZE_Y = 22
debug = False
graphic = False
liner = True
control = True
antiBan = False
soloGame = {'left': 653, 'top': 200 - 67 , 'width': 706, 'height': 746 + 67}
low = np.array([0,0,110])
high = np.array([255,255,255])

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
    if(type == Tetramino.J):
        answer = [[1, 1, 0],[0, 1, 0],[0 ,1, 0]]
    if(type == Tetramino.L):
        answer = [[0, 1, 0],[0, 1, 0],[1 ,1, 0]]
    if(type == Tetramino. O):
        answer = [[1, 1],[1, 1]]
    if(type == Tetramino.Z):
        answer = [[0, 1, 0],[1, 1, 0],[1 ,0, 0]]
    if(type == Tetramino.T):
        answer = [[0, 1, 0],[1, 1, 0],[0 ,1, 0 ]]
    if(type == Tetramino.S):
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



def addNewTetramino(fild, i, a, newFigure):
    try:
        #print(getMatrix(newFigure))
        matrix = np.rot90(getMatrix(newFigure), -a)
        buff = fild
        if(i == -1 and any(matrix[0][temp] == 1 for temp in len((matrix[0])))):
            return None

        for h in range(0,SIZE_Y):
            for x in range(0, len(matrix)):
                for y in range(0, len((matrix[0]))):
                    if(matrix[x][y] == 1):
                        if(h + y + 1 == SIZE_Y):
                            for xC in range(0, len(matrix)):
                                for yC in range(0, len(matrix[0])):
                                    if (matrix[xC][yC] == 1):
                                        buff[h + yC][i + xC] = 1;
                                        if(i + xC < 0):
                                            return None
                            return buff

                        if (buff[h + y + 1][i + x] == 1):
                            for xC in range(0, len(matrix)):
                                for yC in range(0, len(matrix[0])):
                                    if (matrix[xC][yC] == 1):
                                        buff[h + yC][i + xC] = 1;
                                        if(i + xC < 0):
                                            return None
                            return buff
    except:
        return None

def testLineDelete(fild):
    lineSum = 0
    for h in range(SIZE_Y - 1, -1, -1):
        if(all(fild[h][x] == 1 for x in range(0, SIZE_X))):
            lineSum += 1
            for hc in range(h - 1, -1, -1):
                for x in range(0, SIZE_X):
                    fild[hc + 1][x] = 0
                    fild[hc + 1][x] = fild[hc][x]
    return [fild, lineSum]



def genAllfilds(fild, newFigure):
    answer = []
    print(newFigure)
    for i in range(-3 ,12):
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
    for y in range(0, SIZE_Y):
        for x in range(0, SIZE_X):
            if(fild[y][x] == 1):
                closer[x] = 1
                sum += (y - 10)**3 * kh

            if (fild[y][x] == 0 and closer[x] == 1):
                sum -= kd * (SIZE_Y - y + 1)
    sum += fildAndLines[1] * kLines
    sum -= np.sum(np.sum(fild)) * kCount
    return sum


def nothing(x):
 pass

if(debug):
    cv2.namedWindow('image')
    cv2.createTrackbar('H_H','image',0,255,nothing)
    cv2.createTrackbar('H_L','image',0,255,nothing)
    cv2.createTrackbar('S_H','image',0,255,nothing)
    cv2.createTrackbar('S_L','image',0,255,nothing)
    cv2.createTrackbar('V_H','image',0,255,nothing)
    cv2.createTrackbar('V_L','image',0,255,nothing)

def getScreenShotHSV():
    screenShot = mss().grab(soloGame)
    img = np.array(Image.frombytes('RGB', (screenShot.width, screenShot.height), screenShot.rgb, ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


def openCvGetFild():
    img = getScreenShotHSV()
    soloGame = {'left': 653, 'top': 200 - 67, 'width': 706, 'height': 746 + 67}
    gameFild = img[80:(746+67), 175:511]
    fildWeight = 511 - 175
    fildHeight = 746 - 80 + 67
    #hold = img[106:207, 4:175]
    #next = img[0:746, 175:511]
    if (debug):
        high[0] = cv2.getTrackbarPos('H_H', 'image')
        high[1] = cv2.getTrackbarPos('S_H', 'image')
        high[2] = cv2.getTrackbarPos('V_H', 'image')
        low[0] = cv2.getTrackbarPos('H_L', 'image')
        low[1] = cv2.getTrackbarPos('S_L', 'image')
        low[2] = cv2.getTrackbarPos('V_L', 'image')

    gameFildBin = cv2.inRange(gameFild, low, high)
    #holdBin = cv2.inRange(hold, low, high)
    #nextBin = cv2.inRange(next, low, high)

    fildList = []
    for i in range(0, SIZE_Y):
        fildList.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for y in range(0, SIZE_Y):
        for x in range(0, SIZE_X):
            a = 0
            if (areaWhitePercent(gameFildBin[y * fildHeight // SIZE_Y:(y + 1) * fildHeight // SIZE_Y,
                                 x * fildWeight // SIZE_X:(x + 1) * fildWeight // SIZE_X]) > 50):
                fildList[y][x] = 1

    if (graphic):
        if (liner):
            for i in range(0, fildWeight, fildWeight // SIZE_X):
                cv2.line(gameFildBin, (i, 0), (i, fildHeight), (255), thickness=1)
            for i in range(0, fildHeight, fildHeight // SIZE_Y):
                cv2.line(gameFildBin, (0, i), (fildWeight, i), (255), thickness=1)
        #cv2.imshow('test', img)
        cv2.imshow('filg', gameFild)
        #cv2.imshow('hold', hold)
        #cv2.imshow('next', next)
        cv2.imshow('gameFildBin', gameFildBin)
        #cv2.imshow('holdBin', holdBin)
        #cv2.imshow('nextBin', nextBin)
    return fildList

# while( False):
#     fildList = openCvGetFild()
#     for i in range(0, SIZE_X):
#         fildList[0][i] = 0
#         fildList[1][i] = 0
#         fildList[2][i] = 0
#         fildList[3][i] = 0
#     print(q(fildList))
#     time.sleep(1)



while True :# True:
    fildList = openCvGetFild()
    newFigure = newTetramino(fildList)

    #print(newFigure)



    for i in range(0, SIZE_X):
        fildList[0][i] = 0
        fildList[1][i] = 0
        fildList[2][i] = 0
        fildList[3][i] = 0
        fildList[4][i] = 0

    if(newFigure[0] != Tetramino.No):
        allPos = genAllfilds(fildList.copy(), newFigure[0])
        optimum = allPos[0]
        maximum = -99999999999999


        for i in allPos:
            answ = q(testLineDelete(i[2]))
            if(answ > maximum):
                maximum = answ
                optimum = i

        #print(optimum[0], optimum[1])

        # print("\n\n\n\n")
        # for a in allPos:
        #     print("\n\n\n\n")
        #     print(a[0], a[1], q(a[2]))
        #     for i in a[2]:
        #         print(i)
        # print("\n\n\n\n")
        # for i in optimum[2]:
        #     print(i)
        # print(optimum[0], optimum[1], q(optimum[2]))
        #
        # for i in testLineDelete(optimum[2]):
        #     print(i)

        if(control):


            print("a", optimum[1] )
            print(optimum[0] , newFigure[1])

            time.sleep(0.01)
            for i in range(optimum[1]):
                pyautogui.press('z')
                if (antiBan):
                    time.sleep(uniform(0.1, 0.3))
                #time.sleep(0.1)


            if(optimum[0] - newFigure[1] > 0):
                print("r",optimum[0] - newFigure[1])
                for i in range(optimum[0] - newFigure[1]):
                    pyautogui.press('right')
                    if (antiBan):
                        time.sleep(uniform(0.1, 0.3))
                    print("right")
                    #time.sleep(0.1)

            if (optimum[0] - newFigure[1] < 0):
                print("l", newFigure[1] - optimum[0])
                for i in range(newFigure[1] - optimum[0]):
                    pyautogui.press('left')
                    if(antiBan):
                        time.sleep(uniform(0.1, 0.3 ))
                    print("left")
                    #

            pyautogui.press(' ')
            if(antiBan):
                time.sleep(uniform(0.4, 0.5))
        #pyautogui.press('down', 10)

    if cv2.waitKey(33) & 0xFF in (ord('q'),27,):
        break

fildList = []
for i in range(0, SIZE_Y):
    fildList.append([0,0,0,0,0,0,0,0,0,0])


fildList = addNewTetramino(fildList, 2, 1, Tetramino.I)
#fildList = addNewTetramino(fildList, 0, 1, Tetramino.I)
new = copy.deepcopy(fildList)
answ = genAllfilds(new, Tetramino.I)

for i in fildList:
    print(i)

#
# print("\n\n\n\n")
# for a in answ:
#     for i in a[2]:
#         print(i)
#     print(q(a[2]),"\n\n\n\n" )
#
# print(id(fildList), id(new) )
