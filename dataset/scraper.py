import requests, random
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import math
import uuid
import json

def getpixXY(lat, lon, z):
    sinLat = math.sin(lat*math.pi/180)
    x = ((lon+180)/360)*256*(2**z)
    y = (0.5 - math.log((1 + sinLat) / (1 - sinLat)) / (4 * math.pi)) * 256 * (2**z)
    return x, y

def getlatlon(x, y, z):
    lon = (x/256)/(2**z) * 360  - 180
    epi = math.e**math.pi
    epi2 = epi**2
    sinLat = (epi2 - epi**(y*(2**(-z-6))))/(epi**(y*(2**(-z-6))) + epi2)
    lat = math.asin(sinLat)*180/math.pi
    return lat, lon

def get_tile(lat, lon, zoom, disp = False, counter = 5):
    if(counter == 0):
        # print(f"Failed query {lat} {lon} {zoom}")
        return None
    if(counter < 5):
        time.sleep(0.1 * random.randint(0,5))
    
    mapw = 1024
    maph = 1024+96
    url = 'https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/{},{}/{}?mapSize={},{}&key={}'
    key = 'TODO: Add Bing Maps API Key Here'

    url_zoom = url.format(lat, lon, zoom, maph, maph, key)
    response = requests.get(url_zoom)
    try:
        img = Image.open(BytesIO(response.content))
    except:
        return get_tile(lat, lon, zoom, counter = counter-1)
        
    diff = (maph - mapw) // 2
    cropped = img.crop((diff, diff, maph-diff, maph-diff))
    if(disp):
        plt.imshow(cropped)
        plt.show() 
    return cropped

# check if more than 20% of the pixels have value (245, 242, 237)
def check_white(im):
    thresh = 256 
    white = 0
    notwhite = 0
    if(not im):
        return True
    
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if im.getpixel((i, j)) == (245, 242, 237): #or im.getpixel((i, j)) == (172, 199, 242):
                white += 1
                if(white > thresh):
                    return True
            else:
                notwhite += 1
                if(notwhite > thresh):
                    return False
    return False


def get_stitched(lat, lon, zoom):
    mapw = 1024
    maph = 1024+96
    
    x, y = getpixXY(lat, lon, zoom)
    x1 = x - mapw/4
    y1 = y - mapw/4
    x2 = x + mapw/4
    y2 = y + mapw/4

    lattl, lontl = getlatlon(x1, y1, zoom)
    lattr, lontr = getlatlon(x2, y1, zoom)
    latbl, lonbl = getlatlon(x1, y2, zoom)
    latbr, lonbr = getlatlon(x2, y2, zoom)


    cropped1 = get_tile(lattl, lontl, zoom+1)
    if(not cropped1):
        return None
    cropped2 = get_tile(lattr, lontr, zoom+1)
    if(not cropped2):
        return None
    cropped3 = get_tile(latbl, lonbl, zoom+1)
    if(not cropped3):
        return None
    cropped4 = get_tile(latbr, lonbr, zoom+1)
    if(not cropped4):
        return None

    stitch = Image.new("RGB", (2*mapw, 2*mapw), "white")
    stitch.paste(cropped1, (0,0))
    stitch.paste(cropped2, (mapw,0))
    stitch.paste(cropped3, (0,mapw))
    stitch.paste(cropped4, (mapw,mapw))
    return stitch

import random


def get_random_coords(max_count):
    coords = []
    for i in range(max_count):
        # 5% - 20, 20% - 19, 100% - 13
        # start with x random locations
        # Loop through taking max res available for each
        # Once desired num have been reached for certain classes skip
        # - 5% = 7.5k/0.05/17 = 8823.52941
        
        lat = random.uniform(-66.0, 66.0)
        lon = random.uniform(-180.0, 180.0)
        coords.append((lat, lon))
    return coords

def get_urban_coords(max_count):
    jitter = 0.1
    data = json.load(open('major_cities.json'))
    coords = []
    for i in range(max_count):
        index = random.randint(0, len(data)-1)
        lat = float(data[index]['latitude']) + jitter * (random.random()-0.5)
        lon = float(data[index]['longitude']) + jitter * (random.random()-0.5)
        coords.append((lat, lon))
    return coords
    
import os
from multiprocessing import Pool
import time
import math
import shutil

def scrape_bing(coord):
    lat = coord[0]
    lon = coord[1]
    fill_status = get_fill_status()
    mz = get_max_zoom(lat, lon, fill_status)
    if(mz == 0):
        return
    query_pyramid(lat, lon, mz)
    
def get_fill_status():
    global gl20, gl19, gl13
    if urban:
        return [False, False, True]
    if(not gl20 and len(os.listdir('./bing_20/')) > nps/17):
        gl20 = True
    if(not gl19 and len(os.listdir('./bing_19/')) > nps/16):
        gl19 = True
    if(not gl13 and len(os.listdir('./bing_13/')) > nps/10):
        gl13 = True
    return [gl20, gl19, gl13]
    
def query_pyramid(lat, lon, mz):
    folder_name = uuid.uuid4()

    parent_folder = f'bing_{mz}'
    if urban:
        parent_folder = 'bing_urban'

    os.makedirs(f'./{parent_folder}/{folder_name}/')

    # Save meta-data
    with open(f'./{parent_folder}/{folder_name}/metadata.txt', 'w') as w:
        w.write(f"Latitude: {lat}\nLongitude: {lon}")

    for i in range(mz-1, 2, -1): #Allows zooms from level 4 to mz bc of +1 in stitch function
        im = get_stitched(lat, lon, i)
        if(not im):
            # print(f"clearing {lat} {lon} at {i} in {mz}")
            shutil.rmtree(f'./{parent_folder}/{folder_name}/')
            return
        im.save(f"./{parent_folder}/{folder_name}/{i+1 if i>8 else '0'+str(i+1)}.jpg")

    
def get_max_zoom(lat, lon, filled):
    if(not filled[0] and not check_white(get_tile(lat, lon, 20))):
        return 20
    elif(not filled[1] and not check_white(get_tile(lat, lon, 19))):
        return 19
    elif(not filled[2] and not check_white(get_tile(lat, lon, 13))):
        return 13
    return 0

gl13 = False
gl19 = False
gl20 = False

urban = False
nps = 500000 #num_per_stack

if not urban:
    coords = get_random_coords(int(nps/0.02/17*2))
else:
    coords = get_urban_coords(int(nps/17*1.2))

    
start_time = time.time()
with Pool(28) as pool:
    pool.map(scrape_bing, coords)
print(time.time() - start_time)