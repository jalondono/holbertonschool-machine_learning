#!/usr/bin/env python3
"""  How many by rocket? """
import requests


if __name__ == '__main__':

    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)
    launches = r.json()
    rocket_dict = {}

    for i, launch in enumerate(launches):
        rocket_id = launch["rocket"]
        rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(
            rocket_id)
        rocket = requests.get(rocket_url).json()["name"]
        if rocket in rocket_dict.keys():
            rocket_dict[rocket] += 1
        else:
            rocket_dict[rocket] = 1
    rockets = sorted(rocket_dict.items(), key=lambda kv: kv[0])
    rockets = sorted(rockets, key=lambda kv: kv[1], reverse=True)
    for rocket in rockets:
        print(*rocket, sep=": ")
