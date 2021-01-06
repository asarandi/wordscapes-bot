#!/usr/bin/env python3

import glob
import pyautogui
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv
import os, sys
import time


letters = {}
englishWords = set()


def loadWords():
    with open("dictionary.txt") as fp:
        res = set(fp.read().split())
        fp.close()
    return res


def findWords(data: str) -> [str]:
    global englishWords
    dataCounter = Counter(list(data))
    dataLen = sum(dataCounter.values())
    def f(w: str) -> bool:
        return len(w) > 2 and len(w) <= dataLen and len(Counter(list(w)) - dataCounter) == 0
    return list(filter(lambda x: f(x), englishWords))


def letterContours():
    global letters
    font = ImageFont.truetype("KeepCalm-Medium.ttf", 128)  # font size
    for i in range(26):
        c = chr(i + ord('A'))
        img = np.zeros((256,256,3), np.uint8)                   # image size
        image = Image.fromarray(img) #new("1", (72, 72), 0)
        draw = ImageDraw.Draw(image)
        draw.text((32, 32), c, font=font, fill=(255,255,255))
        img = np.array(image.crop(image.getbbox()))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(img, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        letters[c] = contours[0]


def detectLetters(img: Image) -> {}:
    global letters
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    img = cv.GaussianBlur(img, (3, 7), 0)
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    mask = np.zeros(img.shape, np.uint8)
    cv.circle(mask, (176, 590), 110, 1, -1)    # center coords, radius, color, thickness
    img = cv.bitwise_and(img, img, mask=mask)
    img = cv.bitwise_not(img, img, mask=mask)
#    cv.imshow("as", img)
#    cv.waitKey()
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    res = []
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
        if h < 38 or w < 6: continue
        score, letter = 1, '_'
        for k, v in letters.items():
            s = cv.matchShapes(c, v, cv.CONTOURS_MATCH_I1, 0.0)
            if s < score:
                score, letter = s, k
        res.append((letter, x+(w//2), y+(h//2)))
    return res        

def joinLetters(lst: []) -> str:
    return ''.join([x[0] for x in lst]).lower()



def scrcpy():
    cmd = """killall scrcpy ; \
            nohup scrcpy \
                --stay-awake \
                --turn-screen-off \
                --always-on-top \
                --window-borderless \
                --window-x 0 \
                --window-y 0 \
                --max-size 768 \
                --window-height 768 \
                --lock-video-orientation 0 \
                >/dev/null 2>&1 &"""
    os.system(cmd)                

#scrcpy()
letterContours()
englishWords = loadWords()
img = pyautogui.screenshot(region=(0,0,353,768)).convert("RGB")
letters = detectLetters(img)
for c in letters:
    print("letter", c)
word = joinLetters(letters)
words = findWords(word)
words = sorted(words)
words.sort(key=lambda x: len(x))
print(word)
print(words)



