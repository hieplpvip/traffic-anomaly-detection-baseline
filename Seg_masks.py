import json
import pathlib
import time
from typing import Dict

import tqdm

VALID_NAMES = ['car', 'bus', 'truck']


def seg_masks(frame: Dict) -> None:
    filename = pathlib.Path(frame['filename'])
    mask_txt = repo_path / 'Masks' / filename.parent.relative_to(repo_path / 'ori_images')
    mask_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(mask_txt.with_suffix('.txt'), 'a') as output:
        for bound in frame['objects']:
            if bound['name'] in VALID_NAMES:
                a = -1
                b = bound['relative_coordinates']['center_x'] * 800
                c = bound['relative_coordinates']['center_y'] * 410
                d = bound['relative_coordinates']['width'] * 800
                e = bound['relative_coordinates']['height'] * 410
                f = bound['confidence']
                g = -1
                h = -1
                j = -1
                line = [filename.stem, a, b, c, d, e, f, g, h, j]
                for l in line:
                    output.write(str(l) + ',')
                output.write('\n')


if __name__ == '__main__':
    start_time = time.perf_counter()
    repo_path = pathlib.Path(__file__).resolve().parent
    with open(repo_path / 'Masks' / 'part1.json', 'r') as f:
        D = json.load(f)

    for frame in tqdm.tqdm(D):
        seg_masks(frame)

    end_time = time.perf_counter()
    res = time.strftime('%Hh:%Mm:%Ss', time.gmtime(end_time - start_time))
    print(f'Finished in {res}')
