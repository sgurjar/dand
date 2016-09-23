download OSM XML
sandiego openstreet map data from
https://mapzen.com/data/metro-extracts/#san-diego-california

downloaded file compressed with bz2 format (san-diego_california.osm.bz2),
you can use terminal to decompress it with following command-

  bunzip2 san-diego_california.osm.bz2

we are parsing 2 type of tag <node> and <way>
* Node represents a place: https://wiki.openstreetmap.org/wiki/Node
* Way represents a shape using nodes, could be a street, highway
etc. https://wiki.openstreetmap.org/wiki/Way

OSM XML format os documented here
https://wiki.openstreetmap.org/wiki/OSM_XML


<node> looks like folllwing

  <node changeset="8697293" id="365318169" lat="32.7152558"
          lon="-117.149052" timestamp="2011-07-11T20:13:06Z" uid="109570"
          user="Vincent Broman" version="2">
    <tag k="name" v="Jewel Box" />
    <tag k="source" v="SanGIS Business Sites Public Domain (http://www.sangis.org/)" />
    <tag k="amenity" v="bar" />
    <tag k="addr:city" v="San Diego" />
    <tag k="is_in:city" v="San Diego" />
    <tag k="addr:street" v="Broadway" />
    <tag k="is_in:state" v="California" />
    <tag k="addr:country" v="US" />
    <tag k="addr:postcode" v="92101" />
    <tag k="is_in:country" v="United States of America" />
    <tag k="addr:housenumber" v="1600" />
    <tag k="is_in:state_code" v="CA" />
    <tag k="is_in:country_code" v="US" />
  </node>

* We convert this xml node into json format as following, this is done
in shape_element method of data.py.
* process_map (in data.py) calls shape_element for each toplevel xml
element.
* Here we are using iterparse method of xml.etree.ElementTree, since
iterparse method doesnot load whole xml in memory.
@see https://docs.python.org/2/library/xml.etree.elementtree.html#xml.etree.ElementTree.iterparse

  {
    "_type": "node",
    "name": "Jewel Box",
    "created": {
      "changeset": "8697293",
      "version": "2",
      "user": "Vincent Broman",
      "timestamp": "2011-07-11T20:13:06Z",
      "uid": "109570"
    },
    "pos": [
      32.7152558,
      -117.149052
    ],
    "amenity": "bar",
    "source": "SanGIS Business Sites Public Domain (http://www.sangis.org/)",
    "address": {
      "city": "San Diego",
      "street": "Broadway",
      "housenumber": "1600",
      "postcode": "92101",
      "country": "US"
    },
    "id": "365318169",
    "is_in": {
      "city": "San Diego",
      "state": "California",
      "state_code": "CA",
      "country_code": "US",
      "country": "United States of America"
    }
  }

Similary, <way> xml elements are in following format:

  <way changeset="33846226" id="5938355" timestamp="2015-09-07T04:14:29Z" uid="48060" user="Sat" version="6">
    <nd ref="1429663703" />
    <nd ref="3732391609" />
    <nd ref="1429663812" />
    <nd ref="3732391614" />
    <nd ref="48857412" />
    <tag k="name" v="Esplendente Boulevard" />
    <tag k="highway" v="residential" />
    <tag k="tiger:cfcc" v="A41" />
    <tag k="tiger:tlid" v="195321131" />
    <tag k="tiger:county" v="San Diego, CA" />
    <tag k="tiger:source" v="tiger_import_dch_v0.6_20070809" />
    <tag k="tiger:reviewed" v="no" />
    <tag k="tiger:zip_left" v="92124" />
    <tag k="tiger:name_base" v="Esplendente" />
    <tag k="tiger:name_type" v="Blvd" />
    <tag k="tiger:separated" v="no" />
    <tag k="tiger:zip_right" v="92124" />
  </way>

We convert this to in the following json format (using same function
shape_element in data.py)

  {
    "_type": "way",
    "node_refs": [
      "1429663703",
      "3732391609",
      "1429663812",
      "3732391614",
      "48857412"
    ],
    "name": "Esplendente Boulevard",
    "created": {
      "changeset": "33846226",
      "version": "6",
      "user": "Sat",
      "timestamp": "2015-09-07T04:14:29Z",
      "uid": "48060"
    },
    "tiger": {
      "separated": "no",
      "name_base": "Esplendente",
      "zip_left": "92124",
      "tlid": "195321131",
      "cfcc": "A41",
      "reviewed": "no",
      "county": "San Diego, CA",
      "source": "tiger_import_dch_v0.6_20070809",
      "name_type": "Blvd",
      "zip_right": "92124"
    },
    "id": "5938355",
    "highway": "residential"
  }

process_map method of data.py takes san-diego_california.osm (xml)
file as input and generates equivalent san-diego_california.osm.json
file. san-diego_california.osm.json file contains one line for each
<node> or <way> xml element in json format. In above json examples,
indentations are for readbility purpose and not required.

So a json document

  {
    "The Bourne Identity': {
      'year': '2002',
      'month': 'June',
      'day': '14'
     }
  }

is same as all written in one line

  { "The Bourne Identity': {'year': '2002', 'month': 'June', 'day': '14' } }


also we put a "_type" attribute for each node or way element, which
is "_type": "node" or "_type": "way", so later when we query we can
diffrentiate between node and way.

Now we need to import this in mongodb, we use tool that comes with mongo
call mongoimport, I ran following command

  C:\tools\mongodb\Server\3.2\bin\mongoimport.exe /db:osm /collection:sandiego san-diego_california.osm.json

you dont need to specify whole path to mongoimport.exe, if its in your PATH.

Now in MongoDB database have multiple collections, similar to a relational
database has multiple tables. However, in relational database structure of
table is fixed and data that we insert must follow that schema. However,
in MongoDB collections is a "collection" of json documents and each
document can has as many or as few keys as it needed, no schema is
required.

Once we have our data imported in mongodb we can query it using mongodb
aggregation framework. We could use pymongo python library to execute
queries and process data, or we can run from mongodb shell, which
mongo.exe.

statistics.py and queries.py contains queries that used for the project.

For example, if I were to find which zipcode has most bars in san diego, I could run following queries. Usin mongo shell,

$C:\tools\mongodb\Server\3.2\bin\mongo.exe
2016-07-29T18:08:59.849-0400 I CONTROL  [main] Hotfix KB2731284 or later update is not installed, will zero-out data files
MongoDB shell version: 3.2.7
connecting to: test
> show dbs
local  0.000GB
osm    0.100GB
> use osm
switched to db osm
> db.sandiego.aggregate([ {"$match": {"amenity": {"$eq": "bar"}}}])

  If I run just this, this will give all the json document that has "amenity" attribute set to "bar".

to summarize we need to group, so in aggregate framework output of last command goes as input to next command.

>   db.sandiego.aggregate([
...       {"$match": {"amenity": {"$eq": "bar"}}},
...       {"$group": {"_id": "$address.postcode", "count":{"$sum": 1} }}
...     ])
{ "_id" : null, "count" : 225 }
{ "_id" : "92103", "count" : 6 }
{ "_id" : "92120", "count" : 2 }
{ "_id" : "91941", "count" : 1 }
{ "_id" : "92115", "count" : 2 }
{ "_id" : "92101", "count" : 9 }
{ "_id" : "92118", "count" : 1 }
{ "_id" : "92104", "count" : 10 }
{ "_id" : "91932", "count" : 2 }
{ "_id" : "91942", "count" : 1 }
{ "_id" : "92105", "count" : 1 }
{ "_id" : "91911", "count" : 1 }
{ "_id" : "92102", "count" : 2 }
{ "_id" : "92116", "count" : 2 }
{ "_id" : "91977", "count" : 1 }
{ "_id" : "92107", "count" : 1 }
{ "_id" : "92018", "count" : 1 }

since there are many missing values. Now we want to sort the result by count (descending) and get top 5 results.

db.sandiego.aggregate([
    {"$match": {"amenity": {"$eq": "bar"}}},
    {"$group": {"_id": "$address.postcode", "count":{"$sum": 1} }},
    {"$sort": {"count": -1}},
    {"$limit": 5}
  ])


{"$sort": {"count": -1}}
this says we want to sort by count and -1 says descending, default is ascending.

{"$limit": 5}
says we want to result to limited to 5 documents, top 5.


db.sandiego.aggregate([
     {"$match": {"amenity": {"$eq": "bar"}}},
     {"$group": {"_id": "$address.postcode", "count":{"$sum": 1} }},
     {"$sort": {"count": -1}},
     {"$limit": 5}
   ])

{ "_id" : null, "count" : 225 }
{ "_id" : "92104", "count" : 10 }
{ "_id" : "92101", "count" : 9 }
{ "_id" : "92103", "count" : 6 }
{ "_id" : "92115", "count" : 2 }

again, there are too many missing (null) values.

lets see what street has most bars.


db.sandiego.aggregate([
    {"$match": {"amenity": {"$eq": "bar"}}},
    {"$group": {"_id": "$address.street", "count":{"$sum": 1} }},
    {"$sort": {"count": -1}},
    {"$limit": 5}
  ])

{ "_id" : null, "count" : 221 }
{ "_id" : "University Avenueenue", "count" : 7 }
{ "_id" : "5th Avenueenue", "count" : 5 }
{ "_id" : "El Cajon Boulevard", "count" : 3 }
{ "_id" : "Fern Streetreet", "count" : 3 }

again, there are too many null values.

Reverse Geocoding

We have latitude and longitude is each record, we can use that to find
the address using http://nominatim.openstreetmap.org/reverse api, see
http://wiki.openstreetmap.org/wiki/Nominatim#Reverse_Geocoding for docs.

If you put
http://nominatim.openstreetmap.org/reverse?lat=32.802315&lon=-117.228265&format=json
in the browser, it will return a json document.

and when you put,
http://nominatim.openstreetmap.org/reverse?lat=32.802315&lon=-117.228265&format=xml

it will return xml for same data.

We use this api to get address information for latitude and longitude
and use the street, city and postcode info from here.
