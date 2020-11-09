#!/usr/bin/env python3
""" Insert a document in Python """


def insert_school(mongo_collection, **kwargs):
    """ inserts a new document in a collection based on kwargs:
        mongo_collection will be the pymongo collection object
        Returns the new _id
    """
    id_ = mongo_collection.insert_one(kwargs).inserted_id
    return id_
