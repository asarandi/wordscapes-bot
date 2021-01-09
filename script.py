#!/usr/bin/env python3

import os
import sys
import time
from collections import Counter

import cv2 as cv
import numpy as np
import pyautogui
import pytesseract
from PIL import Image


my_linux_config = {
    "scale": 1,
    "window_max_size": 768,
    "window_height": 768,
    "screenshot_region": (0, 0, 353, 768),
    "circle_center": (176, 640),
    "circle_radius": 110,
    "contour_min_height": 20,
    "contour_min_width": 2,
    "rearranged_box_size": 64,
    "rearranged_padding": 8,
}

# mac with retina display, scrcpy flag "--borderless" does not work
# y offset 90 to account for macos top bar + scrcpy window title bar
my_macos_config = {
    "scale": 2,
    "window_max_size": 768,
    "window_height": 768,
    "screenshot_region": (0, 0, 353 * 2, 768 * 2 + 90),
    "circle_center": (176 * 2, 640 * 2 + 90),
    "circle_radius": 110 * 2,
    "contour_min_height": 20 * 2,
    "contour_min_width": 2 * 2,
    "rearranged_box_size": 64 * 2,
    "rearranged_padding": 8 * 2,
}

config = my_linux_config


def detect_letters(img: Image) -> [tuple]:
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    img = cv.threshold(img, 127, 255, cv.THRESH_OTSU)[1]
    mask = np.zeros(img.shape, np.uint8)
    cv.circle(mask, config["circle_center"], config["circle_radius"], 1, -1)
    img = cv.bitwise_and(img, img, mask=mask)

    # check if circle center is mostly dark or white, invert img if needed
    (px, py), size = config["circle_center"], config["rearranged_padding"]
    sample = img[py - size:py + size, px - size:px + size]
    if np.count_nonzero(sample) > sample.size // 2:
        img = cv.bitwise_not(img, img, mask=mask)

    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    box_size, padding = config["rearranged_box_size"], config["rearranged_padding"]
    rearranged = np.zeros((box_size, 8 * box_size), np.uint8)  # enough space for eight letters
    max_y, pos_y, pos_x = 0, padding, padding
    positions = []

    for cnt in contours:
        cx, cy, w, h = cv.boundingRect(cnt)
        if h < config["contour_min_height"] or w < config["contour_min_width"]:
            continue

        positions.append((cx + (w // 2), cy + (h // 4)))

        # re-arrange letters in a circle into a straight line
        rearranged[pos_y:pos_y + h, pos_x:pos_x + w] = img[cy:cy + h, cx:cx + w]
        if pos_y + h > max_y:
            max_y = pos_y + h
        pos_x = pos_x + w + padding

    rearranged = cv.bitwise_not(rearranged, rearranged)[:max_y + padding, :pos_x]
    tes = pytesseract.image_to_string(rearranged, config='--psm 6')
    letters = list(filter(lambda c: c.isupper(), list(tes)))
    if len(letters) != len(positions):
        sys.exit("error detecting letters")
    return list(zip(letters, positions))


def scrcpy():
    cmd = f"""  if pgrep -x scrcpy >/dev/null ;
                then
                    echo 'scrcpy already running'
                else
                    adb shell 'monkey -p com.peoplefun.wordcross 1'
                    nohup scrcpy \
                    --stay-awake \
                    --turn-screen-off \
                    --always-on-top \
                    --window-borderless \
                    --window-x 0 \
                    --window-y 0 \
                    --max-size {config['window_max_size']} \
                    --window-height {config['window_height']} \
                    --lock-video-orientation 0 >/dev/null 2>&1 &
                fi """
    os.system(cmd)


def load_words() -> set:
    with open("words.txt") as fp:
        res = set(fp.read().split())
        fp.close()
    return res


# returns a list of words that can be arranged from scrambled letters
def match_words(word_list: set, letters: [tuple]) -> [str]:
    ctr = Counter([k[0].lower() for k in letters])
    return list(filter(lambda w: len(Counter(list(w)) - ctr) == 0, word_list))


# returns a list of non repeating (x, y) coords
def build_moves(word: str, positions: [tuple]) -> [tuple]:
    res = []
    for k in list(word.upper()):
        for letter, (px, py) in positions:
            if letter == k and (px, py) not in res:
                res.append((px, py))
                break
    return res


if __name__ == "__main__":
    scrcpy()
    all_words = load_words()
    while True:
        x, y = config["circle_center"]
        scale = config["scale"]
        x, y = x // scale, y // scale
        pyautogui.click(x, y)
        time.sleep(2)
        capture = pyautogui.screenshot(region=config["screenshot_region"]).convert("RGB")
        detected = detect_letters(capture)
        for char in detected:
            print(char)
        matches = match_words(all_words, detected)
        matches = sorted(matches)  # alphabetic sort
        matches.sort(key=lambda k: len(k))  # sort shortest to longest
        for matched_word in matches:
            print(matched_word)
            moves = build_moves(matched_word, detected)
            for index, (x, y) in enumerate(moves):
                x, y = x // scale, y // scale
                print("\t", "move", index, (x, y))
                if index == 0:
                    pyautogui.mouseDown(x, y, pyautogui.LEFT)
                elif index == len(moves) - 1:
                    pyautogui.mouseUp(x, y, pyautogui.LEFT)
                else:
                    pyautogui.moveTo(x, y, 0.15)
            time.sleep(0.5)
        time.sleep(12)
