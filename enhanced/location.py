import os

location = 'CN'

def init_location():
    global location
    import json
    import requests
    try:
        location = json.loads(requests.get(f'http://ip-api.com/json').text)['countryCode']
    except Exception as e:
        print(f'Init location error: {e}')
    return

def language_radio(language):
    return '中文' if language=='cn' else 'En'
