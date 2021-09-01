import argparse
import json
import logging
import os
import pathlib
import subprocess
import time
from typing import TextIO

import cv2
import decord
import natsort
import numpy as np
import tqdm


def extract_frames(dest_dir: pathlib.Path, video_path: pathlib.PurePath, root: pathlib.PurePath, time_f: int,
                   ori_images_txt: TextIO, ctx: decord.nd.DECORDContext) -> pathlib.Path:
    """
    :param dest_dir: Directory where the extracted frames will be stored.
    :param video_path: The absolute path of the video.
    :param root: Directory containing the videos to be processed.
    :param time_f: Time frequency.
    :param ori_images_txt: File object where the frames of `video_path` are stored.
    :param ctx: The context to decode the video file.
    :return: Directory containing the stored frames of `video_path`
    """
    pic_path = dest_dir / video_path.relative_to(root).with_suffix("")
    pic_path.mkdir(parents=True, exist_ok=True)
    try:
        vr = decord.VideoReader(os.fspath(video_path), ctx=ctx)
        vr.skip_frames(time_f)
        vr.seek(time_f - 1)
        size = len(vr)
        frames_indices = range(time_f - 1, size, time_f)
        for c in frames_indices:
            img_path = os.fspath(pic_path / (str(c + 1) + '.jpg'))
            cv2.imwrite(img_path, vr.next().asnumpy())
            ori_images_txt.write(img_path + "\n")
    except decord.DECORDError:
        vc = cv2.VideoCapture(os.fspath(video_path))
        if vc.isOpened():
            c = 1
            while vc.grab():
                if c % time_f == 0:
                    img_path = os.fspath(pic_path / (str(c) + '.jpg'))
                    _, frame = vc.retrieve()
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


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(levelname)s - %(module)s - %(funcName)s - %(message)s")
    hdlr = logging.StreamHandler()
    hdlr.setLevel(logging.DEBUG)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
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
    time_f = args.freq

    repo_path = pathlib.Path(__file__).resolve().parent
    dest_dir = repo_path / "ori_images"
    dest_dir_processed = repo_path / "processed_images"
    video_names = frozenset.union(*frozenset(map(lambda e: frozenset(root.rglob("*." + e)), ext)))
    video_names = natsort.natsorted(video_names, alg=natsort.ns.PATH)

    try:
        subprocess.check_output("nvidia-smi")
        ctx = decord.gpu(0)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("NVIDIA GPU device not found.")
        ctx = decord.cpu(0)

    with open(repo_path / "ori_images.txt", "w") as ori_images_txt:
        for video_path in tqdm.tqdm(video_names):
            pic_path = extract_frames(dest_dir, video_path, root, time_f, ori_images_txt, ctx)
            process_frames(dest_dir_processed, pic_path, dest_dir)

    # Store relative paths to root directory.
    with open(repo_path / "dataset.json", "w") as f:
        relative_paths = list(map(lambda v: os.fspath(v.relative_to(root)), video_names))
        json.dump(relative_paths, f)

    end_time = time.perf_counter()
    res = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time))
    print(f"Finished in {res}")
