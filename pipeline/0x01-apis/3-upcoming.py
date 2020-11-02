#!/usr/bin/env python3
"""  What will be next? """
import requests


if __name__ == '__main__':

    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)
    launches = r.json()
    date = float('inf')

    for i, launch in enumerate(launches):
        if date > launch["date_unix"]:
            date = launch["date_unix"]
            index = i

    n = launches[index]["name"]
    d = launches[index]["date_local"]
    r_id = launches[index]["rocket"]
    url = "https://api.spacexdata.com/v4/rockets/{}".format(r_id)
    r_name = requests.get(url).json()["name"]
    lp_id = launches[index]["launchpad"]
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(lp_id)
    lp = requests.get(url).json()
    lp_name = lp["name"]
    lp_loc = lp["locality"]

    print(n + " (" + d + ") " + r_name + " - " + lp_name + " (" + lp_loc + ")")
