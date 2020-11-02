#!/usr/bin/env python3
""" Rate me is you can! """
import sys
import requests
import time


if __name__ == '__main__':

    url = sys.argv[1]
    payload = {'Accept': 'application/vnd.github.v3+json'}
    r = requests.get(url, params=payload)

    if r.status_code == 403:
        limit = r.headers["X-Ratelimit-Reset"]
        x = (int(limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))

    if r.status_code == 200:
        location = r.json()["location"]
        print(location)

    if r.status_code == 404:
        print("Not found")
