#!/usr/bin/env python3
""" contains the  availableShips function"""
import requests


def availableShips(passengerCount):
    """
    create a method that returns the list of ships that
     can hold a given number of passengers:
    :param passengerCount:
    :return:
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []

    while url is not None:
        r = requests.get(url)
        results = r.json()["results"]
        for ship in results:
            p = ship["passengers"]
            p = p.replace(',', '')
            if p.isnumeric() and int(p) >= passengerCount:
                ships.append(ship["name"])
        url = r.json()["next"]
    return ships
