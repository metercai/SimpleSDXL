import os
import json
import requests
import maxminddb
from enhanced.models_hub_host import models_hub_host

location = 'CN'
try:
    public_ip = json.loads(requests.get(f'{models_hub_host}/ip/').text)['ip']
    geoip_path = os.path.abspath(f'./enhanced/geo.ip')
    with maxminddb.open_database(geoip_path) as reader:
        geoinfo = reader.get(public_ip)
        if geoinfo:
            location = geoinfo['country']['iso_code']
except Exception as e:
    print(e)
