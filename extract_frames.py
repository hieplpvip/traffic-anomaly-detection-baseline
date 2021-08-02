import argparse
import os
import pathlib

import cv2
import natsort
import numpy as np
import tqdm

parser = argparse.ArgumentParser(description="Extract frames from directory containing videos.")
parser.add_argument("--root", type=pathlib.Path, help="directory containing videos to be processed",
                    default=pathlib.Path("../Data/test-data/"))
parser.add_argument("--ext", type=str, help="extensions of the videos within the directory to be processed",
                    default="mp4")
args = parser.parse_args()
root = args.root.resolve()
ext = args.ext.split(" ")

repo_path = pathlib.Path(__file__).resolve().parent
dest_dir = repo_path / "ori_images"
video_names = frozenset.union(*frozenset(map(lambda e: frozenset(root.rglob("*." + e)), ext)))
video_names = natsort.natsorted(video_names, alg=natsort.ns.PATH)
videos_folder = []
print("capture videos")
for video_path in tqdm.tqdm(video_names):
    pic_path = dest_dir / video_path.relative_to(root).with_suffix("")
    pic_path.mkdir(parents=True, exist_ok=True)
    videos_folder.append(pic_path)
    vc = cv2.VideoCapture(os.fspath(video_path))
    c = 1
    if vc.isOpened():
        timeF = 100
        while vc.grab():
            _, frame = vc.retrieve()
            if c % timeF == 0:
                cv2.imwrite(os.fspath(pic_path / (str(c) + '.jpg')), frame)
            c += 1
            cv2.waitKey(1)
        vc.release()

dest_dir_processed = repo_path / "processed_images"
print("average images")
for video_path in tqdm.tqdm(videos_folder):
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
