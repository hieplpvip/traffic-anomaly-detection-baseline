#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import os
import pathlib

import numpy as np

from vid_utils import *

# In[2]:

repo_path = pathlib.Path(__file__).resolve().parent
with open(repo_path / 'result.json', 'r') as f:
    D = json.load(f)

#  ## Extract Objects

# In[3]:

All_Cords = extract_objects(D)

# ## Extract Cases

# In[4]:

AT = extract_cases(All_Cords)

# ## Detect Change in Camera

# In[6]:

Base = repo_path / "processed_images"
if (repo_path / "change.npy").exists():
    change_cam, loc, Cstat = np.load(os.fspath(repo_path / "change.npy"), allow_pickle=True)
else:
    change_cam, loc, Cstat = change_detect(os.fspath(Base))
    np.save(os.fspath(repo_path / "change.npy"), [change_cam, loc, Cstat])

# In[7]:

PT = list(set(AT) - set(change_cam))

# ## Case 1: Extract ROI

# In[8]:

if (repo_path / "centers1.npy").exists():
    Centers = np.load(os.fspath(repo_path / "centers1.npy"), allow_pickle=True)
else:
    Centers = extract_roi(PT, All_Cords)
    np.save(os.fspath(repo_path / "centers1.npy"), Centers)

# In[9]:

len(Centers)

# ## Case 1: Extract Bounds

# In[10]:

if (repo_path / "bounds1.npy").exists():
    Bounds = np.load(os.fspath(repo_path / "bounds1.npy"), allow_pickle=True)
else:
    Bounds = extract_bounds(Centers, PT, All_Cords)
    np.save(os.fspath(repo_path / "bounds1.npy"), Bounds)

# In[11]:

len(Bounds)

# ## Case 1: Backtracking

# In[12]:

Base = repo_path / "ori_images"

if (repo_path / "result1.npy").exists():
    Times, Stat = np.load(os.fspath(repo_path / "result1.npy"), allow_pickle=True)
else:
    Times, Stat = backtrack(Bounds, PT, os.fspath(Base))
    np.save(os.fspath(repo_path / "result1.npy"), [Times, Stat])

# ## Case 2: Extract ROI

# In[13]:

if (repo_path / "centers2.npy").exists():
    Centers2 = np.load(os.fspath(repo_path / "centers2.npy"), allow_pickle=True)
else:
    Centers2 = extract_roi1(change_cam, All_Cords, loc)
    np.save(os.fspath(repo_path / "centers2.npy"), Centers2)

# ## Case 2: Extract Bounds

# In[14]:

if (repo_path / "bounds2.npy").exists():
    Bounds2 = np.load(os.fspath(repo_path / "bounds2.npy"), allow_pickle=True)
else:
    Bounds2 = extract_bounds1(Centers2, change_cam, loc, All_Cords)
    np.save(os.fspath(repo_path / "bounds2.npy"), Bounds2)

# ## Case 2: Backtracking

# In[15]:

len(Centers), len(Centers2)

# In[16]:

len(Bounds), len(Bounds2)

# In[17]:

Base = repo_path / "ori_images"

if (repo_path / "result2.npy").exists():
    Times2, Stat2 = np.load(os.fspath(repo_path / "result2.npy"), allow_pickle=True)
else:
    Times2, Stat2 = backtrack1(Bounds2, os.fspath(Base))
    np.save(os.fspath(repo_path / "result2.npy"), [Times2, Stat2])

# In[18]:

Times = {key: val for key, val in Times.items() if val != 999}
Times = {key: val for key, val in Times.items() if val >= 40}

Times2 = {key: val for key, val in Times2.items() if val != 999}
Times2 = {key: val for key, val in Times2.items() if val >= 40}

# In[23]:

with open(repo_path / "Result.txt", "w") as file1:
    for x in Times:
        file1.write('{0:2d} {1:3d} {2:1d}'.format(x, int(Times[x]), 1))
        file1.write("\n")

    for x in Times2:
        file1.write('{0:2d} {1:3d} {2:1d}'.format(x, int(Times2[x]), 1))
        file1.write("\n")
