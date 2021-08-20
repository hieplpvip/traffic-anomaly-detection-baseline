#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import pathlib

import tqdm

# In[6]:

repo_path = pathlib.Path(__file__).resolve().parent
with open(repo_path / 'Masks' / 'part1.json', 'r') as f:
    D = json.load(f)

names = ["car", "bus", "truck"]

for frame in tqdm.tqdm(D):
    filename = pathlib.Path(frame['filename'])
    mask_txt = repo_path / "Masks" / filename.parent.relative_to(repo_path / "ori_images")
    mask_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(mask_txt.with_suffix(".txt"), "a") as file1:
        for bound in frame['objects']:
            if bound['name'] in names:
                # Read the frame number of the filename that is before the underscore.
                frame_n = filename.stem.split("_")[0]
                a = -1
                b = bound['relative_coordinates']['center_x'] * 800
                c = bound['relative_coordinates']['center_y'] * 410
                d = bound['relative_coordinates']['width'] * 800
                e = bound['relative_coordinates']['height'] * 410
                f = bound['confidence']
                g = -1
                h = -1
                j = -1
                line = [frame_n, a, b, c, d, e, f, g, h, j]
                for l in line:
                    file1.write(str(l) + ",")
                file1.write("\n")
