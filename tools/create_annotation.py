import os
import random

import cv2
from tqdm import tqdm

flag_map = {
    "0": 1,
    "1": 0,
    "2": 2,
    "3": 0,
}


def main():
    data_path = "data/operation_surgery/annotations"

    ann = open(os.path.join(data_path, "raw.txt")).readlines()

    flag = 0
    for i in range(len(ann)):
        cur_ann = ann[i].strip().split(" ")
        if len(cur_ann) > 1:
            flag = max([flag_map[x] for x in cur_ann[1:]])
        ann[i] = f"{cur_ann[0]} {flag}\n"

    random.shuffle(ann)
    ann = ann[::25]

    split_num = int(len(ann) * 0.8 + 0.5)

    open(os.path.join(data_path, "train.txt"), "w").writelines(ann[:split_num])
    open(os.path.join(data_path, "val.txt"), "w").writelines(ann[split_num:])


if __name__ == "__main__":
    main()
