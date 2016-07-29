#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymongo import MongoClient

mongodb_url='mongodb://localhost:27017'
client = MongoClient(mongodb_url)
db = client.osm

def print_heading(h):
    print "\n",h
    print len(h)*"="

# Version
##########
buildInfo = db.command('buildInfo')
print_heading('MongoDB Server Version')
print buildInfo['version']

# Database stats
################
dbstats = db.command('dbstats')
print_heading('Database Statistics')
print 'name:', dbstats['db']
print 'number of collections:', dbstats['collections']
print 'number of objects:',dbstats['objects']
print 'total amount of space (in bytes):', dbstats['storageSize']

# Collection stats
##################
colstats = db.command('collstats','sandiego')
print_heading('Collection Statistics')
print 'name:', colstats['ns']
print 'number of objects:', colstats['count']
print 'size of all documents (in bytes):', colstats['size']
print 'total amount of space (in bytes):', colstats['storageSize']

# Distinct Users
################
# print distinct users
# first group, groups all distinct uid
# second group counts such groups. _id=None, we need to could all rows.
nuid = db.sandiego.aggregate([
        {"$group": {"_id": "$created.uid"}},
        {"$group": {"_id": None, "count": {"$sum": 1}}}
    ])

print_heading('Dataset Statistics')
for x in nuid:
    print "number of unique users:", x['count']

# number of node and way
########################
count_by_type = db.sandiego.aggregate([
        {"$group": {"_id": "$_type", "count": {"$sum": 1}}}
    ])

for x in count_by_type:
    print 'number of %s:' % x['_id'], x['count']

# top 10 contributors
#####################
k = 10
topk = db.sandiego.aggregate([
        {"$group": {"_id": "$created.user", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": k}
    ])

print_heading('Top %s Contributors' % k)
for x in topk:
    print '%s:' % x['_id'], x['count']

# top 10 amenity
#####################
k = 10
print_heading('Top %s by Amenity' % k)
count_by_amenity = db.sandiego.aggregate([
        {"$match": {"amenity": {"$ne": None}}}, # ignore nulls
        {"$group": {"_id": "$amenity", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": k}
    ])

for x in count_by_amenity:
    print '%s:' % x['_id'], x['count']

client.close()