import json
import os
import pathlib
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Union, Tuple, Any, Iterable

import cv2
import natsort
import numpy as np
from kneed import KneeLocator
from scipy.signal import savgol_filter
from skimage import metrics
from sklearn.cluster import KMeans

__all__ = ['kmeans_clu', 'reject_outliers', 'check_continual', 'nms', 'extract_objects', 'extract_cases', 'extract_roi',
           'extract_roi1', 'extract_bounds', 'extract_bounds1', 'change_detect', 'backtrack', 'backtrack1']
REPO_PATH = pathlib.Path(__file__).resolve().parent
# Typing hints
Relative_Coordinates = Mapping[str, float]
Objects = List[Mapping[str, Union[int, str, Relative_Coordinates, float]]]
Darknet_JSON = Sequence[Mapping[str, Union[int, str, Objects]]]

# Return value datatype
Vid_Cords = List[np.ndarray]
All_Videos_Cords = List[Vid_Cords]

PT = List[np.ndarray]
Centers = Union[np.ndarray, Iterable, int, float, tuple, dict, List[List[List[np.ndarray]]]]
Centersf = List[Centers]
Cam_Change = List[int]
Loc = List[int]
Times = Dict[Union[int, Any], Union[int, np.ndarray, float, complex]]
All_Statfn = List[List[Union[int, np.ndarray, float, complex]]]
Bounds = List[List[Union[Union[np.ndarray, int], Any]]]
Bounds1 = List[List[Union[int, Any]]]
All_Stat3 = List[List[Union[np.ndarray, int, float, complex]]]


# --------------------------------------------------------------------------------------------------


def get_videos() -> List[str]:
    with open(REPO_PATH / 'dataset.json', 'r') as f:
        videos = json.load(f)
        return videos


VIDEOS = get_videos()


def kmeans_clu(data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, List[int], KMeans]:
    kmeans = KMeans(n_clusters=k, max_iter=10000, n_init=10).fit(data)  ##Apply k-means clustering
    labels = kmeans.labels_
    clu_centres = kmeans.cluster_centers_
    z = [np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)]
    return clu_centres, labels, z, kmeans


def reject_outliers(data: Union[np.ndarray, Iterable]) -> np.ndarray:
    m = 2
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return np.array(filtered)


def check_continual(stat: np.ndarray, k: int) -> bool:
    found = False
    inx = (stat > 0.5).astype(int)
    for ix in range(0, len(inx) - k, k):
        if np.min((inx[ix:ix + k])) == 1:
            found = True
    return found


def nms(boxes: np.ndarray, overlap_thresh: float) -> Union[np.ndarray, List[float]]:
    # if there are no boxes, return an empty list
    if not boxes.any():
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0] - (boxes[:, 3] / 2)
    y1 = boxes[:, 1] - (boxes[:, 2] / 2)
    x2 = boxes[:, 0] + (boxes[:, 3] / 2)
    y2 = boxes[:, 1] + (boxes[:, 2] / 2)

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        #         print(overlap)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    #     print(pick)
    return boxes[pick]


def extract_objects(darknet_json: Darknet_JSON) -> All_Videos_Cords:
    base = REPO_PATH / "ori_images"
    frames_folders = frozenset(map(lambda e: e.parent, base.rglob("*.jpg")))
    frames_folders = natsort.natsorted(frames_folders, alg=natsort.ns.PATH)
    c = 0
    valid_list = ('bus', 'truck', 'person', 'car', 'boat')
    all_cords = list()

    for frames_folder in frames_folders:
        vid_cords = list()
        frames = frames_folder.glob("*.jpg")
        for frame in frames:
            frame_number = int(frame.stem)

            if frame_number % 100 == 0:

                frame_cords = list()

                if (len(darknet_json[c]['objects'])) != 0:
                    for bound in darknet_json[c]['objects']:
                        if bound['name'] in valid_list:
                            x = bound['relative_coordinates']['center_x'] * 400
                            y = bound['relative_coordinates']['center_y'] * 410
                            h = bound['relative_coordinates']['height'] * 410
                            w = bound['relative_coordinates']['width'] * 400
                            f = c
                            if w * h > 16 and w > 4 and h > 4:
                                frame_cords.append([x, y, h, w, f])

                c += 1

                if (len(darknet_json[c]['objects'])) != 0:
                    for bound in darknet_json[c]['objects']:
                        if bound['name'] in valid_list:
                            x = bound['relative_coordinates']['center_x'] * 400 + 200
                            y = bound['relative_coordinates']['center_y'] * 410
                            h = bound['relative_coordinates']['height'] * 410
                            w = bound['relative_coordinates']['width'] * 400
                            f = c
                            if w * h > 16 and w > 4 and h > 4:
                                frame_cords.append([x, y, h, w, f])

                c += 1

                if (len(darknet_json[c]['objects'])) != 0:
                    for bound in darknet_json[c]['objects']:
                        if bound['name'] in valid_list:
                            x = bound['relative_coordinates']['center_x'] * 400 + 400
                            y = bound['relative_coordinates']['center_y'] * 410
                            h = bound['relative_coordinates']['height'] * 410
                            w = bound['relative_coordinates']['width'] * 400
                            f = c
                            if w * h > 16 and w > 4 and h > 4:
                                frame_cords.append([x, y, h, w, f])

                c += 1

                frame_cords = nms(np.array(frame_cords), 0.9)
                vid_cords.append(frame_cords)

        all_cords.append(vid_cords)

    return all_cords


def extract_cases(all_cords: All_Videos_Cords) -> PT:
    maxes = list()
    for vid_cords in all_cords:
        vid_cords = np.array(vid_cords, dtype=object)
        at = [item for frame_cords in vid_cords for item in frame_cords]
        at = np.array(np.array(at)).reshape(-1, 5)

        if at.shape[0] > 10:
            y = []
            K = range(1, 10)
            for k in K:
                km = KMeans(n_clusters=k, max_iter=10000, n_init=10)
                km = km.fit(at[:, 0:4])
                y.append(km.inertia_)

            x = range(1, len(y) + 1)

            kn = KneeLocator(x, y, curve='convex', direction='decreasing')
            k = kn.knee + 1

            if k is None:
                k = 10

            center, labels, z, kmeans = kmeans_clu(at[:, 0:4], k)
            max_count = Counter(labels)

            if at.shape[0] > 1:
                mx = max_count[max(max_count, key=max_count.get)]
            else:
                mx = 0
        else:
            mx = 0
        maxes.append(mx)

    pt = list(np.where(np.array(maxes) > 20)[0] + 1)
    return pt


def extract_roi(pt: PT, all_cords: All_Videos_Cords) -> Centers:
    centers = list()
    mas_dir = REPO_PATH / "Masks" / "Mas"
    mas_npys = natsort.natsorted(iter(mas_dir.rglob("*.npy")), alg=natsort.ns.PATH)
    for mas_npy in mas_npys:
        mask = np.load(mas_npy)
        t = np.array(all_cords[mas_npy.relative_to(mas_dir)], dtype=object)
        at = [item for sublist in t for item in sublist]
        at = np.array(np.array(at)).reshape(-1, 5)
        if at.shape[0] > 20:
            y = []
            for k in range(1, 10):
                km = KMeans(n_clusters=k, max_iter=10000, n_init=10)
                km = km.fit(at[:, 0:4])
                y.append(km.inertia_)

            x = range(1, len(y) + 1)

            kn = KneeLocator(x, y, curve='convex', direction='decreasing')
            k = kn.knee + 1

            if k is None:
                k = 10

            center, lb, z, kmeans = kmeans_clu(at[:, 0:4], k)
            vid_cent = list()
            for cent in center:
                y_m = np.max((0, int(cent[1]) - 10))
                y_ma = np.min((410, int(cent[1]) + 10))
                x_m = np.max((0, int(cent[0]) - 10))
                x_ma = np.min((800, int(cent[0]) + 10))
                if np.max(mask[y_m:y_ma, x_m:x_ma]) == 1:
                    vid_cent.append(cent)
            centers.append(vid_cent)

    return centers


def extract_roi1(cam_change: Cam_Change, all_cords: All_Videos_Cords, loc: Loc) -> Centersf:
    centersf = list()
    for i in range(0, len(cam_change)):
        centers = list()
        t = np.array(all_cords[cam_change[i] - 1])
        at = [item for sublist in t for item in sublist]
        at = np.array(np.array(at)).reshape(-1, 5)
        imx = np.where(at[:, 4] >= 10 * loc[i])[0][0]
        at = at[0:imx, :]

        mask = np.load(os.fspath(REPO_PATH / "Masks/Mas/" / pathlib.PurePath(VIDEOS[i - 1]).with_suffix(".npy")))

        if at.shape[0] > 10:
            y = []
            for k in range(1, 10):
                km = KMeans(n_clusters=k, max_iter=10000, n_init=10)
                km = km.fit(at[:, 0:4])
                y.append(km.inertia_)

            x = range(1, len(y) + 1)
            kn = KneeLocator(x, y, curve='convex', direction='decreasing')
            k = kn.knee + 1

            if k is None:
                k = 10

            center, lb, z, kmeans = kmeans_clu(at[:, 0:4], k)
            vid_cent = list()
            for cent in center:
                y_m = np.max((0, int(cent[1]) - 10))
                y_ma = np.min((410, int(cent[1]) + 10))
                x_m = np.max((0, int(cent[0]) - 10))
                x_ma = np.min((800, int(cent[0]) + 10))
                if np.max(mask[y_m:y_ma, x_m:x_ma]) == 1:
                    vid_cent.append(cent)

            centers.append(vid_cent)
        centersf.append(centers)
    return centersf


def extract_bounds(centers: Centers, pt: PT, all_cords: All_Videos_Cords) -> Bounds:
    count = 0
    bounds = list()
    for centro in centers:

        t = np.array(all_cords[-1 + pt[count]], dtype=object)
        at = [item for sublist in t for item in sublist]
        at = np.array(np.array(at)).reshape(-1, 5)

        count2 = 1
        for c in centro:
            found = False
            idx = 0
            while not found:
                if idx < len(at):
                    if np.abs(at[idx, 0] - c[0]) < 25 and np.abs(at[idx, 1] - c[1]) < 25 and at[idx, 2] < 100 and int(
                            at[idx, 2]) > 7 and int(at[idx, 3]) > 7:
                        bounds.append([at[idx, 0:], pt[count], count2])
                        found = True
                else:
                    found = True
                idx += 1
            count2 += 1
        count += 1

    return bounds


def extract_bounds1(centers: Centers, cam_change: Cam_Change, loc: Loc, all_cords: All_Videos_Cords) -> Bounds1:
    bounds = list()
    for i in range(0, len(cam_change)):
        for centro in centers[i]:

            t = np.array(all_cords[cam_change[i] - 1])
            at = [item for sublist in t for item in sublist]
            at = np.array(np.array(at)).reshape(-1, 5)
            imx = np.where(at[:, 4] >= 10 * loc[i])[0][0]
            at = at[0:imx, :]

            count2 = 1
            for c in centro:
                found = False
                idx = 0
                while not found:
                    if idx < len(at):
                        if np.abs(at[idx, 0] - c[0]) < 25 and np.abs(at[idx, 1] - c[1]) < 25 \
                                and at[idx, 2] < 100 and int(at[idx, 2]) > 6 and int(at[idx, 3]) > 6:
                            bounds.append([at[idx, 0:], cam_change[i], count2])
                            found = True
                    else:
                        found = True
                    idx += 1
                count2 += 1

    return bounds


def change_detect(base: os.PathLike[str]) -> Tuple[Cam_Change, Loc, All_Stat3]:
    base = pathlib.Path(base)
    frames_folders = frozenset(map(lambda e: e.parent, base.rglob("*.jpg")))
    frames_folders = natsort.natsorted(frames_folders, alg=natsort.ns.PATH)
    all_stat3 = list()

    for frames_folder in frames_folders:
        stat = list()
        frames = iter(map(lambda f: os.fspath(f), frames_folder.rglob("*.jpg")))
        frames = natsort.natsorted(frames, alg=natsort.ns.PATH)
        for idx in range(1, len(frames) - 10, 10):
            img0 = cv2.imread(frames[idx])
            img1 = cv2.imread(frames[idx + 10])

            is_zero = np.sum(img1) == 0 or np.sum(img0) == 0
            sad = 0.99 if is_zero else metrics.structural_similarity(img0, img1, multichannel=True, win_size=3)

            stat.append(np.max((0, sad)))
        all_stat3.append(stat)

    cam_change = list()
    loc = list()
    for idx, stat3 in enumerate(all_stat3, start=1):
        if np.mean(stat3) - np.min(stat3) > 0.06:
            cam_change.append(idx)
            loc.append(-2 + np.where(np.array(stat3) == np.min(stat3))[0][0])

    return cam_change, loc, all_stat3


def backtrack(all_bounds, pt: PT, base: os.PathLike[str]) -> Tuple[Times, All_Statfn]:
    base = pathlib.Path(base)
    all_statfn = list()
    data = []

    for bounds in all_bounds:

        if bounds[1] in pt:
            base2 = base / pathlib.PurePath(VIDEOS[(bounds[1] - 1)]).with_suffix("")

            files = natsort.natsorted(frozenset(base2.glob("*.jpg")), alg=natsort.ns.PATH)

            stat = list()
            back_len = int(bounds[0][4] * 100)
            x = int(bounds[0][0])
            y = int(bounds[0][1])
            h = int(bounds[0][2] / 2)
            w = int(bounds[0][3] / 2)

            last_frame = int(files[- 1].stem)

            if back_len > last_frame:
                back_len = int(back_len / 10)

            if back_len <= last_frame:
                img0 = cv2.imread(os.fspath(base2 / (str(back_len) + ".jpg")))[y - h:y + h, x - w:x + w]
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

                data.append({
                    'relative_path': VIDEOS[(all_bounds[bounds][1]) - 1].split('.')[0],
                    'img0': str(back_len) + ".jpg",
                    'imgs1': [],
                    'stat': []
                })

                for idx in range(np.min((26750, 2 * back_len)), 90, -5):
                    if idx % 10 == 0 and idx <= last_frame:
                        img1 = cv2.imread(os.fspath(base2 / (str(idx) + ".jpg")))[y - h:y + h, x - w:x + w]
                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                        ssim = metrics.structural_similarity(img0, img1, multichannel=True, win_size=3)

                        data[-1]['imgs1'].append(str(idx) + ".jpg")
                        data[-1]["stat"].append(ssim)

                        stat.append(ssim)

                for idx in range(0, len(stat) - 35):
                    stat[idx] = np.max((stat[idx:idx + 35]))
                    data[-1]["stat"][idx] = np.max((stat[idx:idx + 35]))

                all_statfn.append(stat)

    times = {}
    count = 0
    for stat in all_statfn:
        times[all_bounds[count][1]] = 999
        count += 1

    count = 0
    for stat in all_statfn:
        image_counter = 0

        if np.max(stat) > 0.5 and np.min(stat) < 0.65:
            stat = savgol_filter(stat, 21, 1)
            nstat = (list(reversed(stat)) - min(stat)) / (max(stat) - min(stat))
            found = check_continual(nstat, 150)
            if found:
                times[all_bounds[count][1]] = np.min((times[all_bounds[count][1]], np.min(
                    ((np.where(np.array(nstat) >= 0.4)[0][0]) * 5 / 30, times[all_bounds[count][1]]))))

            image_counter += 1

        count += 1

    return times, all_statfn


def backtrack1(all_bounds: Bounds, base: os.PathLike[str]) -> Tuple[Times, All_Statfn]:
    base = pathlib.Path(base)
    all_statfn = list()
    data = list()

    for bounds in all_bounds:
        base / pathlib.PurePath(VIDEOS[(bounds[1] - 1)]).with_suffix("")

        base2 = base / pathlib.PurePath(VIDEOS[(bounds[1])]).with_suffix("")

        files = natsort.natsorted(frozenset(base2.glob("*.jpg")), alg=natsort.ns.PATH)

        stat = list()
        back_track = int(bounds[0][4] * 100)
        x = int(bounds[0][0])
        y = int(bounds[0][1])
        h = int(bounds[0][2] / 2)
        w = int(bounds[0][3] / 2)

        last_frame = int(files[-1].stem)

        if back_track > last_frame:
            back_track = int(back_track / 10)

        if back_track <= last_frame:
            img0 = cv2.imread(os.fspath(base2 / (str(back_track) + ".jpg")))[y - h:y + h, x - w:x + w]
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

            for idx in range(np.min((26750, 2 * back_track)), 90, -5):
                if idx % 10 == 0:
                    img1 = cv2.imread(os.fspath(base2 / (str(idx) + ".jpg")))[y - h:y + h, x - w:x + w]
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                    data.append({
                        'relative_path': VIDEOS[(bounds[1]) - 1].split('.')[0],
                        'img0': str(back_track) + ".jpg",
                        'imgs1': [],
                        'stat': []
                    })

                    stat.append(metrics.structural_similarity(img0, img1, multichannel=True, win_size=3))

            for idx in range(0, len(stat) - 35):
                stat[idx] = np.max((stat[idx:idx + 35]))
                data[0]["stat"][idx] = np.max((stat[idx:idx + 35]))

            all_statfn.append(stat)

    times = {}
    count = 0
    for stat in all_statfn:
        times[all_bounds[count][1]] = 999

        count += 1

    count = 0
    image_counter = 0

    for stat in all_statfn:
        stat = reject_outliers(stat)
        if np.max(stat) > 0.5 and np.min(stat) < 0.55:
            stat = savgol_filter(stat, 21, 1)

            nstat = (list(reversed(stat)) - min(stat)) / (max(stat) - min(stat))
            found = check_continual(nstat, 150)

            if found:
                if (np.where(np.array(nstat) >= 0.4))[0][0] != 0:
                    times[all_bounds[count][1]] = np.min((times[all_bounds[count][1]], np.min(
                        ((np.where(np.array(nstat) >= 0.5)[0][0]) * 5 / 30, times[all_bounds[count][1]]))))

                image_counter += 1
        count += 1

    return times, all_statfn
