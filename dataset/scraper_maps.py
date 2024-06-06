import requests, random
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import math
import time

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

def get_tile(lat, lon, zoom, disp = False, counter = 5, maptype='Road'):
    if(counter == 0):
        print(f"Failed query {lat} {lon} {zoom}")
        return None
    if(counter < 5):
        time.sleep(0.2 * random.randint(0,5))
    
    mapw = 1024
    maph = 1024+96
    url = 'https://dev.virtualearth.net/REST/v1/Imagery/Map/{}/{},{}/{}?mapSize={},{}&key={}'
    key = 'TODO: Add Bing Maps API Key Here'

    url_zoom = url.format(maptype, lat, lon, zoom, maph, maph, key)
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

import os
from multiprocessing import Pool
import time
import math
import shutil
import tqdm




def scrape_bing(folder):
    with open(os.path.join(root, dataset, folder, 'metadata.txt'), 'r') as f:
        content = f.read()
        try:
            latitude = float(content.split('\n')[0].split(':')[1])
            longitude = float(content.split('\n')[1].split(':')[1])
            params = (latitude, longitude, ds, folder)
        except:
            print(f"error in {folder}")
            print(content)
            return

    lat = params[0]
    lon = params[1]
    mz = params[2]
    uuid = params[3]
    query_pyramid(lat, lon, mz, uuid)


def query_pyramid(lat, lon, mz, uuid):
    for i in range(9, mz, 2): #Allows zooms from level 4 to mz bc of +1 in stitch function
        
        # if alr exists skip
        if(os.path.exists(f"{root}/bing_{sys.argv[1]}/{uuid}/{i+1 if i>8 else '0'+str(i+1)}_map.jpg")):
            print("skipping", f"{root}/bing_{sys.argv[1]}/{uuid}/{i+1 if i>8 else '0'+str(i+1)}_map.jpg")
            continue

        im = get_stitched(lat, lon, i)
        if(not im):
            continue
        else:
            print(f"writing {lat} {lon} at {i} in {sys.argv[1]} from {uuid}")

        im.save(f"{root}/bing_{sys.argv[1]}/{uuid}/{i+1 if i>8 else '0'+str(i+1)}_map.jpg")

# read in mz from sys args
import sys
if sys.argv[1] == 'urban':
    ds = 20
else:
    ds = int(sys.argv[1])


root = '/scratch/bbut/bing_datasets'
dataset = f"bing_{sys.argv[1]}"



print("starting", dataset)
with Pool(28) as p:
    p.map(scrape_bing, os.listdir(os.path.join(root, dataset)))
print("finished", dataset)
    
    