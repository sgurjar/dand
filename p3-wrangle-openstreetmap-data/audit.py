"""
Your task in this exercise has two steps:

- audit the OSMFILE and change the variable 'mapping' to reflect the changes needed to fix
    the unexpected street types to the appropriate ones in the expected list.
    You have to add mappings only for the actual problems you find in this OSMFILE,
    not a generalized solution, since that may and will depend on the particular area you are auditing.
- write the update_name function, to actually fix the street name.
    The function takes a string with street name as an argument and should return the fixed name
    We have provided a simple test so that you see what exactly is expected
"""
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

#OSMFILE = "example.osm"
OSMFILE = "san-diego_california_sample.osm"
#OSMFILE = "san-diego_california.osm"

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
phone_number_re = re.compile(r'^\+1|[-()]|\s')

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road",
            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = {"Av"     : "Avenue",
           "Ave"    : "Avenue",
           "Ave."   : "Avenue",
           "Blvd"   : "Boulevard",
           "Blvd."  : "Boulevard",
           "Ct"     : "Court",
           "Ctr"    : "Center",
           "Dr"     : "Drive",
           "Dr."    : "Drive",
           "Ln"     : "Lane",
           "Pk"     : "Parkway",
           "Pl"     : "Place",
           "Prky"   : "Parkway",
           "Rd"     : "Road",
           "Rd."    : "Road",
           "St"     : "Street",
           "St."    : "Street",
           "street" : "Street",
           "Wy"     : "Way" }


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


def update_name(name, mapping):
    for k in mapping:
        pos = name.find(k)
        if pos > -1:
            return name[:pos] + mapping[k] + name[pos+len(k):]

    return name

"""
clean addr:state value from the data
this code is specific to San Diego data.
    san-diego_california.osm

$grep addr:state san-diego_california.osm | sort | uniq -c
   2993                 <tag k="addr:state" v="CA"/>
      1                 <tag k="addr:state" v="Ca"/>
     16                 <tag k="addr:state" v="California"/>
      9                 <tag k="addr:state" v="ca"/>


We make it consistent and convert to CA
"""
def is_state(tag):
    return tag.get('k')=="addr:state"

def update_state(state):
    v = state.strip().upper()
    if v=="CALIFORNIA":
        return "CA"
    else:
        return v

def is_phone(tag):
    """<tag k="phone" v="(619) 849-3100"/>"""
    return tag.get('k')=="phone"

def update_phone(phone):
    return phone_number_re.sub('', phone.strip())

def is_postcode(tag):
    return tag.get('k')=="addr:postcode"

def update_postcode(phone):
    valid_postcode_chars = '0123456789-'
    return ''.join([c for c in phone if c in valid_postcode_chars])

def clean_value(tag):
    """returns clean value of <tag>"""
    value = tag.get('v')
    if is_street_name(tag):
        return update_name(value, mapping)
    elif is_state(tag):
        return update_state(value)
    elif is_phone(tag):
        return update_phone(value)
    elif is_postcode(tag):
        return update_postcode(value)
    else:
        return value

def test():
    st_types = audit(OSMFILE)
    assert len(st_types) == 3
    pprint.pprint(dict(st_types))

    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            print name, "=>", better_name
            if name == "West Lexington St.":
                assert better_name == "West Lexington Street"
            if name == "Baldwin Rd.":
                assert better_name == "Baldwin Road"


if __name__ == '__main__':
    test()
