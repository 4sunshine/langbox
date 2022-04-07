import os.path
import random
import urllib.request
from duckduckgo_images_api import search


def image_download(request, target_dir, max_trials=10):
    results = search(request)
    count = len(results['results'])
    if count == 0:
        return None
    for i in range(min(max_trials, count)):
        #index = random.randint(0, count - 1)
        try:
            address = results['results'][i]['image']
            filename = address.split('/')[-1]
            target_path = os.path.join(target_dir, filename)
            urllib.request.urlretrieve(address, target_path)
            return target_path
        except Exception as e:
            print(e)
            continue
    return None
