#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymongo import MongoClient, GEOSPHERE
import sys
import pprint
from urllib import urlencode
import urllib2
import json

mongodb_url='mongodb://localhost:27017'
client = MongoClient(mongodb_url)
db = client.osm


def create_geospatial_index():
        # pos: [latitude, longitude]
    return db.sandiego.create_index([("pos", GEOSPHERE)])


def query(*q):
    return db.sandiego.aggregate(list(q))

def set_proxy():
    proxy = urllib2.ProxyHandler({'http': 'one.proxy.att.com:8080'})
    opener = urllib2.build_opener(proxy)
    urllib2.install_opener(opener)



def nominatim_reverse(**kwargs):
    """get address from latitude and longitude using Nominatim Reverse Geocoding API
       Response is in following format:
        {
            "display_name": "Double Standard Kitchenetta & Bar, G Street, Hillcrest, San Diego, San Diego County, California, 92101, United States of America",
            "place_id": "2623007603",
            "lon": "-117.1590468",
            "boundingbox": [ "32.7123092", "32.7125092", "-117.1591468", "-117.1589468" ],
            "osm_type": "node",
            "licence": "Data \u00a9 OpenStreetMap contributors, ODbL 1.0. http://www.openstreetmap.org/copyright",
            "osm_id": "4284785901",
            "lat": "32.7124092",
            "address": {
                "city": "San Diego",
                "bar": "Double Standard Kitchenetta & Bar",
                "country": "United States of America",
                "county": "San Diego County",
                "suburb": "Hillcrest",
                "state": "California",
                "postcode": "92101",
                "country_code": "us",
                "road": "G Street"
            }
        }
    """
    url = 'http://nominatim.openstreetmap.org/reverse'
    lat = kwargs['lat']; lon = kwargs['lon']
    data = {}
    data['format'] = 'json'
    data['lat'] = lat; data['lon'] = lon

    resp = urllib2.urlopen(url+"?"+urlencode(data))
    try:
        data = json.loads(resp.read())
    finally:
        resp.close()

    if 'error' in data:
        raise Exception(data, kwargs)
    else:
        # so next query will match exactly
        data['lat'] = lat; data['lon'] = lon
        return data

def reverse_geocoding(**kwargs):
    """find address by latitude, longitude in db.geo collection
    if not found use nominatim_reverse to get it and store in db.geo
    for next time."""
    latitude  = kwargs['lat']
    longitude = kwargs['lon']
    address = db.geo.find_one( { 'lat': latitude, 'lon': longitude } )
    if address is None: # not found
        address = nominatim_reverse(lat=latitude, lon=longitude) # get from api
        db.geo.insert_one(address)
    return address



def topk_amenities(k=10):
    """top k amenity"""
    count_by_amenity = query(
            { "$match": { "amenity": { "$ne": None } } }, # ignore nulls
            { "$group": { "_id": "$amenity", "count": { "$sum": 1 } } },
            { "$sort" : { "count": -1 } },
            { "$limit": k } )

    return [(x['_id'], x['count']) for x in count_by_amenity]



def topk_cities_with_parking(k=10):
    """which city has most parking"""
    topk = query(
            { "$match"  : { "amenity": { "$eq": "parking" } } }
           ,{ "$project": { "_id": 0, "city": "$address.city", "type": "$_type" } }
           ,{ "$group"  : { "_id": { "city": "$city", "type": "$type" }, "count": { "$sum": 1 } } }
           ,{ "$sort"   : { "count": -1 } }
           ,{ "$limit"  : k } )

    return [(x['_id']['type'], x['_id'].get('city'),x['count']) for x in topk]



def print_topk_amenities():
    for amenity, count in topk_amenities():
        print amenity, count



def print_topk_cities_with_parking():
    for type, city, count in topk_cities_with_parking():
        print type, city, count



def topk_highway_values(k=10):
    r = query(
            { "$match": { "highway": { "$ne": None } } }
           ,{ "$project": { "highway": 1, "_id": 0, "_type": 1 } }
           ,{ "$group": { "_id": { "highway": "$highway", "type": "$_type" }, "count": { "$sum": 1 } } }
           ,{ "$sort": { "count": -1 } }
           ,{ "$limit": k }
        )

    return [(x['count'], x['_id']['type'], x['_id']['highway']) for x in r]

def populate_bars():
    """postcode, street and city have many missing values
    use reverse_geocoding to get address of node.pos"""

    # drop bars collection if it exists
    db.bars.drop()

    r = query(
            { "$match"  : { "$and": [ { "_type": { "$eq": "node" } }, { "amenity": { "$eq": "bar" } } ] } }
           ,{ "$project": { "pos" : 1, "name": 1, "_id": 0 } }
        )

    bars=[]
    for x in r: #lat=32.7124092&lon=-117.1590469
        lat, lon = x['pos']
        address = reverse_geocoding(lat=lat, lon=lon)['address']
        """
        "address": {
            "city": "San Diego",
            "bar": "Double Standard Kitchenetta & Bar",
            "country": "United States of America",
            "county": "San Diego County",
            "suburb": "Hillcrest",
            "state": "California",
            "postcode": "92101",
            "country_code": "us",
            "road": "G Street"
        }
        """
        bars.append({
            'name'    : x['name'],
            'street'  : address.get('road',address.get('pedestrian')),
            'city'    : address.get('city',address.get('town')),
            'postcode': address['postcode'][:5] # only store first 5 digits
            })


    db.bars.insert_many(bars)

def topk_bars_by(by, k=10):
    r = db.bars.aggregate([
            { "$group": { "_id" : "$"+by, "count": {"$sum": 1} } }
           ,{ "$sort": { "count": -1 } }
           ,{ "$limit": k }
        ])
    return [ (x['count'],x['_id']) for x in r]

def print_topk_highway_values():
    for count, type, highway in topk_highway_values():
        print '%10d %-4s %s' % (count, type, highway)


def foo():
    r = query(
            { "$project": { "_id": 0, "pos": { "$ne": [ "$pos", None ] }, "_type": 1 } }
           ,{ "$group": { "_id": {"type": "$_type", "pos": "$pos"}, "count": { "$sum": 1 } } }
        )
    for x in r:
        print x['count'], x['_id']['type'], x['_id']['pos']

if __name__=='__main__':
    #print_topk_amenities()
    #print_topk_cities_with_parking()
    #print_topk_highway_values()
    #foo()
    #create_geospatial_index()
    #for k,v in db.sandiego.index_information().items():
    #    print k, v
    #bars()
    # lat, lon to address
    # http://nominatim.openstreetmap.org/reverse?lat=32.7124092&lon=-117.1590469
    # http://wiki.openstreetmap.org/wiki/Nominatim#Reverse_Geocoding
    #set_proxy()
    #print nominatim_reverse(lat='32.7124092', lon='-117.1590469')
    # [32.746804, -117.1607671]
    # lat=32.7124092&lon=-117.1590469

    populate_bars()

    print '\ncity'
    for count, city in topk_bars_by('city'):
        print count, city

    print '\npostcodes'
    for count, postcode in topk_bars_by('postcode'):
        print count, postcode

    print '\nstreet'
    for count, street in topk_bars_by('street'):
        print count, street

