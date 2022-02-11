#!/usr/bin/env python3

import sys
import json
import time
import hashlib
import requests
import datetime
from os import path

headers = {
    "Content-Type": "text/plain;charset=UTF-8",
    "Connection": "keep-alive",
    "Accept": "*/*",
    "User-Agent": "Wordscapes/1578 CFNetwork/978.0.7 Darwin/18.7.0",
    "Accept-Language": "en-us",
}

now = datetime.datetime.now().strftime("%s")

config = {
    "app":     "wx",                 # word cross
    "v":       "1578",               # app version
    "rt":      "f",                  # real time?
    "p":       "ios",                # platform
    "te":      "0000000000000000",   # team id ? FIXME
    "ti":      now,                  # time
    "usna":    "user",               # user name ? FIXME
    "uspc":    "00",                 # ? FIXME
    "crt":     "1",                  # ?
    "crs":     "0000000000000000",   # ? FIXME
    "fbid":    "",                   # facebook id
    "us":      "0000000000000000",   # user id FIXME
    "sp":      now,                  # started playing
    "fsp":     "-1",
    "host":    "https://word-cross.appspot.com",
    "secret1": "Snc0ChYyFjEo2YH2KxaEIdjkekCnKhA",
    "secret2": "284682323"
}

def add_checksum(url: str, post_data: str) -> str:
    data = url.split("?", 2)[1] + post_data + config["us"] + config["secret1"] + config["secret2"]
    checksum = hashlib.md5(data.encode("utf8")).hexdigest()
    return url + "&ch=" + checksum


def send_message(msg: str):
    cfg = config
    cfg["ti"] = datetime.datetime.now().strftime("%s")
    fields = ["app", "v", "rt", "p", "te", "ti", "usna", "uspc", "crt", "crs", "fbid", "us", "sp", "fsp"]
    url = cfg["host"] + "/seme?" + "&".join([f"{k}={cfg[k]}" for k in fields])
    url = add_checksum(url, msg)
    try:
        r = requests.post(url, data=msg, headers=headers)
        print("send_message()", r.status_code)
    except:
        pass


def read_messages() -> []:
    cfg = config
    now = datetime.datetime.now().strftime("%s")
    cfg["tsp"], cfg["msp"], cfg["csp"] = now, now, now
    fields = ["app", "v", "rt", "p", "te", "tsp", "msp", "csp", "us", "sp", "fsp"]
    url = cfg["host"] + "/reteu?" + "&".join([f"{k}={cfg[k]}" for k in fields])
    url = add_checksum(url, "x")
    try:
        r = requests.post(url, data="x", headers=headers)
        if r.status_code == 200:
            data = r.json()
            if data and ("ch" in data):
                return data["ch"]
            else:
                return []
        else:
            print("read_messages()", r.status_code)
            return []
    except:
        return []


def load_data() -> {}:
    levels, current = {}, 0
    level_map_file = "ipa/Payload/Wordscapes.app/data/data/level_map.json"

    with open(level_map_file, "r") as fp:
        level_map = json.load(fp)
        fp.close()

    for key_1, value_1 in level_map.items():
        if not (len(key_1) > 2 and key_1[:2] == "WS"):
            continue
        i = int(key_1[2:]) - 1
        if i in (2, 3):
            continue
        for key_2, value_2 in value_1["sets"].items():
            j, level_count = int(key_2[4:]), int(value_2["lc"])
            filename = f"ipa/Payload/Wordscapes.app/data/data/levels/base_level_group_{i}_{j}.json"
            if not path.exists(filename):
                sys.exit("file does not exist", filename)
            with open(filename, "r") as fp:
                data = json.load(fp)
                fp.close()
            for k in range(level_count):
                level = f"level_{k + 1:03d}"
                if level not in data:
                    sys.exit(f"level {level} does not exist in {filename}")
                current += 1
                levels[current] = data[level]
                levels[current]["filename"] = filename
    return levels


def answers(levels: {}, lvl: int) -> str:
    words = sorted([w.lower().split(",")[-1] for w in levels[lvl]["e"]])
    words.sort(key=lambda k: len(k))
    return " ".join(words)


seen_messages = {}
levels = load_data()
while True:
    messages = read_messages()
    print(datetime.datetime.now(), len(messages))
    for message in read_messages():
        if ("ci" not in message) or ("me" not in message) or ("nm" not in message):
            continue
        ci, me, nm = message["ci"], message["me"], message["nm"]
        if ci in seen_messages:
            continue
        seen_messages[ci] = True
        if me.isnumeric() and 1 <= int(me) <= 6000:
            ans = answers(levels, int(me))
            send_message(f"@{nm}, {me}: {ans}")
    time.sleep(120)
