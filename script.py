#!/usr/bin/env python3


import pyautogui
import numpy as np
import cv2 as cv
import os
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

config = {
    "window_max_size": 768,
    "window_max_height": 768,
    "screenshot_region": (0, 0, 353, 768),
    "circle_center": (176, 590),
    "circle_radius": 110,
    "detected_min_height": 38,
    "detected_min_width": 6,
}


def load_words() -> set:
    with open("words.txt") as fp:
        res = set(fp.read().split())
        fp.close()
    return res


def match_words(word_list: set, search: str) -> [str]:
    ctr = Counter(list(search))
    return list(filter(lambda w: len(Counter(list(w)) - ctr) == 0, word_list))


def make_letter_contours() -> {}:
    res = {}
    font = ImageFont.truetype("KeepCalm-Medium.ttf", 128)  # font size
    for i in range(26):
        letter = chr(i + ord('A'))
        image = np.zeros((256, 256, 3), np.uint8)  # image size
        image = Image.fromarray(image)  # new("1", (72, 72), 0)
        draw = ImageDraw.Draw(image)
        draw.text((32, 32), letter, font=font, fill=(255, 255, 255))
        image = np.array(image.crop(image.getbbox()))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(image, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        res[letter] = contours[0]
    return res


def detect_letters(letter_contours: {}, img: Image) -> {}:
    global config
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    img = cv.GaussianBlur(img, (3, 7), 0)
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    mask = np.zeros(img.shape, np.uint8)
    cv.circle(mask, config["circle_center"], config["circle_radius"], 1, -1)
    img = cv.bitwise_and(img, img, mask=mask)
    img = cv.bitwise_not(img, img, mask=mask)
    #    cv.imshow("img", img)
    #    cv.waitKey()
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt1 in contours:
        x, y, w, h = cv.boundingRect(cnt1)
        if h < config["detected_min_height"] or w < config["detected_min_width"]:
            continue
        best_score, best_letter = 1, '_'
        for letter, cnt2 in letter_contours.items():
            score = cv.matchShapes(cnt1, cnt2, cv.CONTOURS_MATCH_I1, 0.0)
            if score < best_score:
                best_score, best_letter = score, letter
        res.append((best_letter, best_score, x + (w // 2), y + (h // 2)))
    return res


def join_letters(lst: [tuple]) -> str:
    return "".join([x[0] for x in lst])


def scrcpy():
    global config
    cmd = f"""
                killall scrcpy;                                 \
                nohup scrcpy                                    \
                    --stay-awake                                \
                    --turn-screen-off                           \
                    --always-on-top                             \
                    --window-borderless                         \
                    --window-x 0                                \
                    --window-y 0                                \
                    --max-size {config['window_max_size']}      \
                    --window-height {config['window_height']}   \
                    --lock-video-orientation 0                  \
                    >/dev/null 2>&1 &
        """
    os.system(cmd)


# scrcpy()
all_letters = make_letter_contours()
all_words = load_words()
capture = pyautogui.screenshot(region=config["screenshot_region"]).convert("RGB")
detected = detect_letters(all_letters, capture)
for char in detected:
    print(char)
results = match_words(all_words, join_letters(detected).lower())
results = sorted(results)
results.sort(key=lambda x: len(x))
print(results)
