import json
import os
import pathlib

import numpy as np
import pandas as pd

from vid_utils import *

repo_path = pathlib.Path(__file__).resolve().parent
with open(repo_path / 'result.json', 'r') as f:
    D = json.load(f)

#  ## Extract Objects

All_Cords = extract_objects(D)

AT = extract_cases(All_Cords)

# ## Detect Change in Camera

Base = repo_path / "processed_images"
if (repo_path / "change.npy").exists():
    change_cam, loc, Cstat = np.load(os.fspath(repo_path / "change.npy"), allow_pickle=True)
else:
    change_cam, loc, Cstat = change_detect(os.fspath(Base))
    np.save(os.fspath(repo_path / "change.npy"), [change_cam, loc, Cstat])

PT = list(set(AT) - set(change_cam))
print("PT", PT)
print("change_cam", change_cam)

# Case 1: Extract ROI

if (repo_path / "centers1.npy").exists():
    Centers = np.load(os.fspath(repo_path / "centers1.npy"), allow_pickle=True)
else:
    Centers = extract_roi(PT, All_Cords)
    np.save(os.fspath(repo_path / "centers1.npy"), Centers)

len(Centers)

# Case 1: Extract Bounds

if (repo_path / "bounds1.npy").exists():
    Bounds = np.load(os.fspath(repo_path / "bounds1.npy"), allow_pickle=True)
else:
    Bounds = extract_bounds(Centers, PT, All_Cords)
    np.save(os.fspath(repo_path / "bounds1.npy"), Bounds)

# Case 1: Backtracking

Base = repo_path / "ori_images"

if (repo_path / "result1.npy").exists():
    Times, Stat = np.load(os.fspath(repo_path / "result1.npy"), allow_pickle=True)
else:
    Times, Stat = backtrack(Bounds, PT, os.fspath(Base))
    np.save(os.fspath(repo_path / "result1.npy"), [Times, Stat])

# Case 2: Extract ROI

if (repo_path / "centers2.npy").exists():
    Centers2 = np.load(os.fspath(repo_path / "centers2.npy"), allow_pickle=True)
else:
    Centers2 = extract_roi1(change_cam, All_Cords, loc)
    np.save(os.fspath(repo_path / "centers2.npy"), Centers2)

# Case 2: Extract Bounds

if (repo_path / "bounds2.npy").exists():
    Bounds2 = np.load(os.fspath(repo_path / "bounds2.npy"), allow_pickle=True)
else:
    Bounds2 = extract_bounds1(Centers2, change_cam, loc, All_Cords)
    np.save(os.fspath(repo_path / "bounds2.npy"), Bounds2)

# Case 2: Backtracking

Base = repo_path / "ori_images"

if (repo_path / "result2.npy").exists():
    Times2, Stat2 = np.load(os.fspath(repo_path / "result2.npy"), allow_pickle=True)
else:
    Times2, Stat2 = backtrack1(Bounds2, os.fspath(Base))
    np.save(os.fspath(repo_path / "result2.npy"), [Times2, Stat2])

Times = {key: val for key, val in Times.items() if val != 999}
Times = {key: val for key, val in Times.items() if val >= 40}

Times2 = {key: val for key, val in Times2.items() if val != 999}
Times2 = {key: val for key, val in Times2.items() if val >= 40}

data = {"video_id": [], "start_second": [], "is_anomaly": []}
for vid, start_second in Times.items():
    data.get("video_id").append(vid)
    data.get("start_second").append(start_second)
    data.get("is_anomaly").append(1)

for vid, start_second in Times2.items():
    data.get("video_id").append(vid)
    data.get("start_second").append(start_second)
    data.get("is_anomaly").append(1)

df = pd.DataFrame(data).sort_values(by=["video_id"])
df.to_csv(repo_path / "Result.csv", index=False)
