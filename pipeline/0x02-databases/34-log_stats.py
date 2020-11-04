#!/usr/bin/env python3
""" Log stats """
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx

    num_doc = logs_collection.count_documents({})
    print("{} logs".format(num_doc))

    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        num_method = logs_collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, num_method))

    filter_path = {"method": "GET", "path": "/status"}
    num_path = logs_collection.count_documents(filter_path)
    print("{} status check".format(num_path))
