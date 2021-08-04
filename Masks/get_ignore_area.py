import pathlib

import natsort
import numpy as np
import tqdm
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label

masks_dir = pathlib.Path(__file__).resolve().parent
masks_txt = iter(map(lambda txt: txt.relative_to(masks_dir), masks_dir.rglob("*.txt")))
masks_txt = natsort.natsorted(masks_txt, alg=natsort.ns.PATH)
mas_dir = masks_dir / "Mas"

count_thred = 0.02
min_area = 500
gass_sigma = 2
score_thred = 0.1

for mask_txt in tqdm.tqdm(masks_txt):
    dt_results_fbf = {}
    with open((masks_dir / mask_txt), 'r') as f:
        for line in f:
            line = line.rstrip()
            word = line.split(',')
            frame = int(word[0])
            x1 = int(float(word[2]))
            y1 = int(float(word[3]))
            tmp_w = int(float(word[4]))
            tmp_h = int(float(word[5]))
            score = float(word[6])
            if frame not in dt_results_fbf:
                dt_results_fbf[frame] = []
            if score > score_thred:
                dt_results_fbf[frame].append([x1, y1, x1 + tmp_w, y1 + tmp_h, score])

    h = 410
    w = 800
    c = 3
    mat = np.zeros((h, w))
    for frame in dt_results_fbf:
        if frame < 18000:
            tmp_score = np.zeros((h, w))

            for box in dt_results_fbf[frame]:
                score = box[4]
                tmp_score[int(float(box[1])):int(float(box[3])), int(float(box[0])):int(float(box[2]))] = np.maximum(
                    score, tmp_score[int(float(box[1])):int(float(box[3])), int(float(box[0])):int(float(box[2]))])

            mat = mat + tmp_score

    mat = mat - np.min(mat)
    mat = mat / np.max(mat)
    mask = mat > count_thred
    mask = label(mask, connectivity=1)
    num = np.max(mask)
    print(num)
    for i in range(1, int(num + 1)):
        if np.sum(mask == i) < min_area:
            mask[mask == i] = 0
    mask = mask > 0
    mask = mask.astype(float)
    k = gaussian_filter(mask, gass_sigma)
    mask = k > count_thred
    mask_npy = mas_dir / mask_txt.with_suffix(".npy")
    mask_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_npy, mask)
