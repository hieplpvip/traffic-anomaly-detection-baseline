import concurrent.futures
import itertools
import os
import pathlib
import time
from typing import TextIO

import cv2
import natsort


def extract_processed_frames(base2: pathlib.Path, frames_folder: pathlib.PurePath, base: pathlib.Path,
                             image_text_file: TextIO) -> None:
    """

    :param base2: Directory where the processed frames will be stored.
    :param frames_folder: The absolute path containing the frames of a video.
    :param base: Directory containing the frames to be processed.
    :param image_text_file: File object where the frames of `frames_folder` are stored.
    :return: None
    """
    new_frames_folder = base2 / frames_folder
    new_frames_folder.mkdir(parents=True, exist_ok=True)
    files = natsort.natsorted(base.rglob("*.jpg"), alg=natsort.ns.PATH)
    for f in files:
        D = cv2.imread(os.fspath(f))
        for i, v in enumerate(range(0, 401, 200), start=1):
            dest = os.fspath(new_frames_folder / (f.stem + "_" + str(i) + ".jpg"))
            cv2.imwrite(dest, D[:, v:v + 400, :])
            image_text_file.write("".join([dest, "\n"]))


start_time = time.perf_counter()
repo_path = pathlib.Path(__file__).resolve().parent
base = repo_path / "processed_images"
base2 = repo_path / "processed_images2"

frames_folders = frozenset(map(lambda e: e.parent.relative_to(base), base.rglob("*.jpg")))
frames_folders = natsort.natsorted(frames_folders, alg=natsort.ns.PATH)
with open(repo_path / "processed_images2.txt",
          "w") as image_text_file, concurrent.futures.ThreadPoolExecutor() as executor:
    executor.starmap(extract_processed_frames, itertools.repeat(base2, len(frames_folders)), frames_folders,
                     itertools.repeat(base, len(frames_folders)),
                     itertools.repeat(image_text_file, len(frames_folders)))

end_time = time.perf_counter()
print(f"Finished in {round(end_time - start_time, 2)} second(s)")
