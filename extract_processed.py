import os
import pathlib
import time

import cv2
import natsort
import tqdm


def extract_processed_frames(base2: pathlib.Path, frames_folder: pathlib.Path, base: pathlib.Path) -> None:
    """
    :param base2: Directory where the processed frames will be stored.
    :param frames_folder: The absolute path containing the frames of a video.
    :param base: Directory containing the frames to be processed.
    :return: None
    """
    new_frames_folder = base2 / frames_folder.relative_to(base)
    new_frames_folder.mkdir(parents=True, exist_ok=True)
    files = natsort.natsorted(frames_folder.glob("*.jpg"), alg=natsort.ns.PATH)
    for f in files:
        D = cv2.imread(os.fspath(f))
        for i, v in enumerate(range(0, 401, 200), start=1):
            dest = os.fspath(new_frames_folder / (f.stem + "_" + str(i) + ".jpg"))
            cv2.imwrite(dest, D[:, v:v + 400, :])


if __name__ == "__main__":
    start_time = time.perf_counter()
    repo_path = pathlib.Path(__file__).resolve().parent
    base = repo_path / "processed_images"
    base2 = repo_path / "processed_images2"

    frames_folders = frozenset(map(lambda e: e.parent, base.rglob("*.jpg")))
    frames_folders = natsort.natsorted(frames_folders, alg=natsort.ns.PATH)
    for frames_folder in tqdm.tqdm(frames_folders):
        print(" ", os.fspath(frames_folder.relative_to(base)))
        extract_processed_frames(base2, frames_folder, base)

    end_time = time.perf_counter()
    res = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time))
    print(f"Finished in {res}")
