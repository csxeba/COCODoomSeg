import os
import json
from collections import defaultdict

import numpy as np
import cv2

import masking


TRAIN_META = "/data/Datasets/cocodoom/run-train.json"
VAL_META = "/data/Datasets/cocodoom/run-val.json"
IMG_ROOT = "/data/Datasets/cocodoom/"


class COCODoomDataset:

    def __init__(self, root, subset):
        self.root = root

        data = json.load(open(os.path.join(root, f"run-{subset}.json")))

        self.index = defaultdict(list)
        for anno in data["annotations"]:
            self.index[anno["image_id"]].append(anno)

        self.image_meta = {meta["id"]: meta for meta in data["images"]}
        self.num_classes = len(data["categories"])

        self.classes = {cat["id"]: i for i, cat in enumerate(data["categories"], start=1)}

        print(f"{subset} num images:", len(data["images"]))
        print(f"{subset} num annos :", len(data["annotations"]))

    def make_sample(self, image_id):
        meta = self.image_meta[image_id]
        image = cv2.imread(os.path.join(self.root, meta["file_name"]))
        mask = np.zeros(image.shape[:2] + (self.num_classes+1,))
        for anno in self.index[image_id]:
            instance_mask = masking.get_mask(anno, image.shape[:2])
            category = self.classes[anno["category_id"]]
            mask[..., category][instance_mask] = 1

        return image, mask

    def stream(self, batch_size, shuffle=True):
        ids = np.array(sorted(self.index))
        N = len(ids)

        while 1:
            if shuffle:
                np.random.shuffle(ids)
            for batch in (ids[start:start+batch_size] for start in range(0, N, batch_size)):
                X, Y = [], []
                for ID in batch:
                    x, y = self.make_sample(ID)
                    X.append(x)
                    Y.append(y)

                yield np.array(X) / 255, np.array(Y)
