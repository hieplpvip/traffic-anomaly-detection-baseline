import os
import pathlib

import cv2
import natsort
import tqdm

repo_path = pathlib.Path(__file__).resolve().parent
base = repo_path / "processed_images"
base2 = repo_path / "processed_images2"

frames_folders = frozenset(map(lambda e: e.parent.relative_to(base), base.rglob("*.jpg")))
frames_folders = natsort.natsorted(frames_folders, alg=natsort.ns.PATH)
with open(repo_path / "processed_images2.txt", "w") as image_text_file:
    for frames_folder in tqdm.tqdm(frames_folders):
        new_frames_folder = base2 / frames_folder
        new_frames_folder.mkdir(parents=True, exist_ok=True)
        files = natsort.natsorted(base.rglob("*.jpg"), alg=natsort.ns.PATH)
        for f in files:
            D = cv2.imread(os.fspath(f))
            for i, v in enumerate(range(0, 401, 200), start=1):
                dest = os.fspath(new_frames_folder / (f.stem + "_" + str(i) + ".jpg"))
                cv2.imwrite(dest, D[:, v:v + 400, :])
                image_text_file.write("".join([dest, "\n"]))
