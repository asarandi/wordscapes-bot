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


# adjust as necessary
my_linux_config = {
    "scale": 1,
    "window_max_size": 768,
    "window_height": 768,
    "screenshot_region": (0, 0, 353, 768),
    "circle_center": (176, 640),
    "shuffle_button": (32, 520),
    "close_piggybank": (285, 210),
    "circle_radius": 109,
    "rearranged_box_size": 64,
    "rearranged_padding": 8,
    "forbid_three_circle": (44, 44 + 22, 736, 736 + 22),
    "letter_contour_dim": (30, 39, 6, 50),
}

config = my_linux_config


def scale_t(t: tuple) -> tuple:
    return tuple(map(lambda f: f * config["scale"], t))


def scale_i(i: int) -> int:
    return i * config["scale"]


def circle_contour() -> np.ndarray:
    r = scale_i(11)
    img = np.zeros((r*4, r*4, 1), np.uint8)
    img = cv.circle(img, (r*2,r*2), r, 255, -1)
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours[0]


def largest_contour(image: np.ndarray) -> np.ndarray:
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    res = contours[0]
    _, _, rw, rh = cv.boundingRect(res)
    for cnt in contours:
        _, _, w, h = cv.boundingRect(cnt)
        if w*h > rw*rh:
            res = cnt
    return res


# does bottom left corner have the "forbid 3" circle
def is_forbid_three(img: Image, compare: np.ndarray) -> bool:
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    binary = cv.threshold(img, 127, 255, cv.THRESH_OTSU)[1]
    x0, x1, y0, y1 = scale_t(config["forbid_three_circle"])
    sub_image = binary[y0 : y1, x0 : x1]
    largest = largest_contour(sub_image)
    if largest is None:
        return False
    score = cv.matchShapes(largest, compare, cv.CONTOURS_MATCH_I1, 0.0)
    return score < 0.05


def detect_letters(img: Image) -> (bool, []):
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    otsu = cv.threshold(img, 127, 255, cv.THRESH_OTSU)[1]
    mask = np.zeros(otsu.shape, np.uint8)
    center, radius = scale_t(config["circle_center"]), scale_i(config["circle_radius"])
    cv.circle(mask, center, radius, 1, -1)
    otsu = cv.bitwise_and(otsu, otsu, mask=mask)
    (px, py), size = center, scale_i(8)
    sample = otsu[py - size:py + size, px - size:px + size]
    if np.count_nonzero(sample) > sample.size // 2:
        otsu = cv.bitwise_not(otsu, otsu, mask=mask)
    all_contours, _ = cv.findContours(otsu, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = []
    min_h, max_h, min_w, max_w = scale_t(config["letter_contour_dim"])
    for cnt in all_contours:
        _, _, w, h = cv.boundingRect(cnt)
        if (min_h <= h <= max_h) and (min_w <= w <= max_w):
            contours.append(cnt)
    if len(contours) not in (6, 7):    # expecting puzzle to have 6 or 7 letters
        return False, []

    box_size = scale_i(config["rearranged_box_size"])
    padding = scale_i(config["rearranged_padding"])
    rearranged = np.zeros((box_size, 8 * box_size), np.uint8)  # enough space for eight letters
    max_y, pos_y, pos_x = 0, padding, padding

    positions = []
    for cnt in contours:
        cx, cy, w, h = cv.boundingRect(cnt)
        positions.append((cx + (w // 2), cy + (h // 4)))

        # re-arrange letters in a circle into a straight line
        rearranged[pos_y:pos_y + h, pos_x:pos_x + w] = otsu[cy:cy + h, cx:cx + w]
        if pos_y + h > max_y:
            max_y = pos_y + h
        pos_x = pos_x + w + padding

    rearranged = cv.bitwise_not(rearranged, rearranged)[:max_y + padding, :pos_x]
    tes = pytesseract.image_to_string(rearranged, config="--psm 13")
    letters = list(filter(lambda c: c.isupper(), list(tes)))
    if len(letters) != len(positions):
        return False, []
    return True, list(zip(letters, positions))


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


def click(xy_t: tuple):
    # macos needs Y offset
    cx, cy = scale_t(xy_t)
    pyautogui.click(cx, cy)


if __name__ == "__main__":
    scrcpy()
    all_words = load_words()
    circle = circle_contour()
    prev_letters = ""

    while True:
        click(config["circle_center"])
        time.sleep(2)

        region = scale_t(config["screenshot_region"])
        capture = pyautogui.screenshot(region=region).convert("RGB")

        success, detected = detect_letters(capture)
        if not success:
            click(config["shuffle_button"])
            click(config["close_piggybank"])
            continue

        letters = ''.join([k[0] for k in detected])
        print(f"letters {letters} previous {prev_letters}")
        if letters == prev_letters:
            click(config["shuffle_button"])
            continue
        prev_letters = letters

        for char in detected:
            print(char)

        matches = match_words(all_words, detected)
        if is_forbid_three(capture, circle):
            matches = list(filter(lambda w: len(w) > 3, matches))
        matches = sorted(matches)               # alphabetic sort
        matches.sort(key=lambda k: len(k))      # sort shortest to longest
        for matched_word in matches:
            print(matched_word)
            moves = build_moves(matched_word, detected)
            for index, (x, y) in enumerate(moves):
                print("\t", "move", index, (x, y))
                if index == 0:
                    pyautogui.mouseDown(x, y, pyautogui.LEFT)
                elif index == len(moves) - 1:
                    pyautogui.mouseUp(x, y, pyautogui.LEFT)
                else:
                    pyautogui.moveTo(x, y, 0.15)
            time.sleep(0.5)
        time.sleep(12)
