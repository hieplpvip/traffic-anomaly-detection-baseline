import argparse
import concurrent.futures
import itertools
import json
import os
import pathlib
import time
from typing import TextIO

import cv2
import natsort
import numpy as np


def extract_frames(dest_dir: pathlib.Path, video_path: pathlib.PurePath, root: pathlib.PurePath, time_f: int,
                   ori_images_txt: TextIO) -> pathlib.Path:
    """
    :param dest_dir: Directory where the extracted frames will be stored.
    :param video_path: The absolute path of the video.
    :param root: Directory containing the videos to be processed.
    :param time_f: Time frequency.
    :param ori_images_txt: File object where the frames of `video_path` are stored.
    :return: Directory containing the stored frames of `video_path`
    """
    vc = cv2.VideoCapture(os.fspath(video_path))
    c = 1
    if vc.isOpened():
        pic_path = dest_dir / video_path.relative_to(root).with_suffix("")
        pic_path.mkdir(parents=True, exist_ok=True)
        while vc.grab():
            _, frame = vc.retrieve()
            if c % time_f == 0:
                img_path = os.fspath(pic_path / (str(c) + '.jpg'))
                cv2.imwrite(img_path, frame)
                ori_images_txt.write(img_path + "\n")

            c += 1
            cv2.waitKey(1)
        vc.release()
        return pic_path


def process_frames(dest_dir_processed: pathlib.Path, video_path: pathlib.Path, dest_dir: pathlib.PurePath) -> None:
    """
    :param dest_dir_processed: Directory where the processed frames will be stored.
    :param video_path: The absolute path containing the frames of a video.
    :param dest_dir: Directory where the extracted frames are stored.
    :return: None
    """
    path_file_number = natsort.natsorted(video_path.glob("*.jpg"), alg=natsort.ns.PATH)
    internal_frame = 4
    start_frame = 100
    nums_frames = len(path_file_number)
    alpha = 0.1
    processed_video_path = dest_dir_processed / video_path.relative_to(dest_dir)
    processed_video_path.mkdir(parents=True, exist_ok=True)

    former_im = cv2.imread(os.fspath(video_path / "100.jpg"))
    for j in range(4, 5):
        internal_frame = 100
        img = cv2.imread(os.fspath(video_path / (str(start_frame) + '.jpg')))
        for n in range(nums_frames):
            now_im = cv2.imread(os.fspath(video_path / (str(n * internal_frame + start_frame) + '.jpg')))
            if now_im is not None:
                filename = str(n * internal_frame + start_frame) + '_' + str(j) + '.jpg'
                if np.mean(np.abs(now_im - former_im)) > 5:
                    img = img * (1 - alpha) + now_im * alpha
                    cv2.imwrite(os.fspath(processed_video_path / filename), img)
                else:
                    cv2.imwrite(os.fspath(processed_video_path / filename), img * 0)
                former_im = now_im


start_time = time.perf_counter()
parser = argparse.ArgumentParser(description="Extract frames from directory containing videos.")
parser.add_argument("--root", type=pathlib.Path, help="directory containing videos to be processed",
                    default=pathlib.Path("../Data/test-data/"))
parser.add_argument("--ext", type=str, help="extensions of the videos within the directory to be processed",
                    default="mp4")
parser.add_argument("--freq", type=int, help="time frequency", default=100)
args = parser.parse_args()

root = args.root.resolve()
ext = args.ext.split(" ")
timeF = args.freq

repo_path = pathlib.Path(__file__).resolve().parent
dest_dir = repo_path / "ori_images"
video_names = frozenset.union(*frozenset(map(lambda e: frozenset(root.rglob("*." + e)), ext)))
video_names = natsort.natsorted(video_names, alg=natsort.ns.PATH)
videos_folder = None
print("capture videos")
with open(repo_path / "ori_images.txt", "w") as ori_images_txt, concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.starmap(extract_frames, itertools.repeat(dest_dir, len(video_names)), video_names,
                               itertools.repeat(root, len(video_names)), itertools.repeat(timeF, len(video_names)),
                               itertools.repeat(ori_images_txt, len(video_names)))
    videos_folder = set(results)

dest_dir_processed = repo_path / "processed_images"
print("average images")
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.starmap(process_frames, itertools.repeat(dest_dir_processed, len(videos_folder)), video_names,
                               itertools.repeat(dest_dir, len(videos_folder)))

# Store relative paths to root directory.
with open(repo_path / "dataset.json", "w") as f:
    relative_paths = list(map(lambda v: os.fspath(v.relative_to(root)), video_names))
    json.dump(relative_paths, f)

end_time = time.perf_counter()
print(f"Finished in {round(end_time - start_time, 2)} second(s)")
