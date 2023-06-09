# -*- coding: utf-8 -*-
"""chicken_breed_rcognizer_data_collections.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hu82cTKZ7B1NOtMVhB582SV9PDKPmLkG

# installing cfscrape
"""

!pip install -U cfscrape

"""# Preparing for a li'l bit of scraping"""

import requests
import bs4  # beautiful soup
# import cfscrape # needed this, as the site was protected 
import numpy as np
import pandas as pd
import pickle
import random

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Dataset/chicken_breed_recognizer

url = "https://www.chickensandmore.com/chicken-breeds"
scraper = cfscrape.create_scraper()
web_page = bs4.BeautifulSoup(scraper.get(url).text, "lxml")

headings = []
for heading in web_page.find_all('h3'): # all the chicken breed names were in h3 tag
    headings.append(heading.text.strip())

len(headings)

type(headings)

headings

""">> so we actually don't need few things, first we don't need the first thing, I mean the first element. And few last elements. However, there still remains a lot of chicken breeds. Should we take names randomly!! Perhaps

# Creating our chicken list

Let's take names randomly
"""

chicken_list = []
rand_idx = random.sample(range(1, 100), 20)    # we'll be taking 20 categories, that's what 20 in the random.sample() function

# another way of doing the same thing
# rng = np.random.default_rng()
# rand_idx = rng.choice(np.arange(1, 100), 20, replace=False)

for i in rand_idx:
    chicken_list.append(headings[i])

chicken_list    # wooho, we have our chicken list, randomly taken

chicken_list = sorted(chicken_list)
chicken_list

with open("chicken_breeds", "wb") as fp:
    pickle.dump(chicken_list, fp)

"""# Setting up for the actual work"""

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext autoreload
# %autoreload
# %matplotlib inline
bs = 8 # batchsize

!pip install -Uqq fastai fastbook nbdev

from fastai import *
from fastbook import *
from fastai.vision.all import *

# load the pickle file 
with open("chicken_breeds", "rb") as fp:
    chicken_list = pickle.load(fp)

chicken_list

for idx, _ in enumerate(chicken_list):
    # print(chicken_list[idx])
    if 'chicken' not in chicken_list[idx]:
        chicken_list[idx] += ' chicken'
chicken_list

images = search_images_ddg(chicken_list[0], max_images=200)
f"No of image => {len(images)} -- One Image Url => {images[0]}"

dest = 'Austra White chicken.jpg'
download_url(images[1], dest, show_progress=False)

image = Image.open(dest)
image.to_thumb(256, 256)

data_path = "data"

if not os.path.exists(data_path):
    os.mkdir(data_path)

for chicken_type in chicken_list:
    dest = os.path.join(data_path, chicken_type)
    if not os.path.exists(dest):
        os.mkdir(dest)
    
    try:
        chicken_images_url = search_images_ddg(chicken_type)
        download_images(dest, urls=chicken_images_url)
    except:
        continue

image_counts = get_image_files(data_path)
image_counts

failed = verify_images(image_counts)
failed

failed.map(Path.unlink)

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.15, seed=11),
    get_y=parent_label,
    item_tfms=Resize(128)
)

doc(parent_label)

dls = dblock.dataloaders(data_path, bs=bs)

dls.train.show_batch(max_n=8, nrows=2)

dls.valid.show_batch(max_n=8, nrows=2)

dblock = dblock.new(item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms()) 
dls = dblock.dataloaders(data_path)
dls.train.show_batch(max_n=8, nrows=2)

torch.save(dls, 'chicken_breeds_dataloader_v0.pkl')

