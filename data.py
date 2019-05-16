import os
import json
from collections import defaultdict

import numpy as np
import cv2

import masking


class COCODoomDataset:

    BACKGROUND_CLASS = 0

    def __init__(self, root, subset, batch_size=16, class_frequency_threshold=0.01, ignored_classes=None):
        self.root = root
        self.batch_size = batch_size

        data = json.load(open(os.path.join(root, f"run-{subset}.json")))

        self.index = defaultdict(list)
        for anno in data["annotations"]:
            self.index[anno["image_id"]].append(anno)
        self.image_meta = {meta["id"]: meta for meta in data["images"]}

        if class_frequency_threshold or ignored_classes is not None:
            class_frequency_threshold = class_frequency_threshold or 0.
            categories, ignores = self._apply_class_frequency_threshold(
                data["annotations"], data["categories"], class_frequency_threshold, ignored_classes)
        else:
            categories = {cat["id"]: cat for cat in data["categories"]}
            ignores = None

        self.ignores = ignores
        self.classes = {}
        self.num_classes = 1
        for ID, category in categories.items():
            if category["ignore"]:
                self.classes[ID] = 0
            else:
                self.classes[ID] = self.num_classes
                self.num_classes += 1

        print(f"{subset} num images :", len(data["images"]))
        print(f"{subset} num annos  :", len(data["annotations"]))
        print(f"{subset} num classes:", self.num_classes)

    @property
    def steps_per_epoch(self):
        return len(self.index) // self.batch_size

    @staticmethod
    def _apply_class_frequency_threshold(annotations, categories, threshold, ignores=None):
        category_index = {cat["id"]: cat for cat in categories}
        class_freqs = defaultdict(int)
        N = 0
        dropped = 0
        if ignores is None:
            ignores = set()

        for anno in annotations:
            class_freqs[anno["category_id"]] += 1
            N += 1

        for ID, freq in class_freqs.items():
            percent = freq / N
            if percent < threshold or ID in ignores:
                ignores.add(ID)
                category_index[ID]["ignore"] = True
                print(f"Ignoring class: {percent:>7.2%} ({category_index[ID]['name']})")
            else:
                category_index[ID]["ignore"] = False

        print(f"Dropped {dropped} classes due to low frequency.")
        return category_index, ignores

    def make_sample(self, image_id):
        meta = self.image_meta[image_id]
        image = cv2.imread(os.path.join(self.root, meta["file_name"]))
        mask = np.zeros(image.shape[:2] + (self.num_classes+1,))
        for anno in self.index[image_id]:
            instance_mask = masking.get_mask(anno, image.shape[:2])
            category = self.classes[anno["category_id"]]
            if category == 0:
                continue
            mask[..., category][instance_mask] = 1
        assert mask[..., 0].sum() == 0
        overlaps = mask.sum(axis=2)[..., None]
        overlaps[overlaps == 0] = 1
        mask /= overlaps
        mask[..., 0] = 1 - mask[..., 1:].sum(axis=2)
        assert np.all(mask[..., 0] >= 0)
        return image, mask

    def stream(self, shuffle=True):
        ids = np.array(sorted(self.index))
        N = len(ids)

        while 1:
            if shuffle:
                np.random.shuffle(ids)
            for batch in (ids[start:start+self.batch_size] for start in range(0, N, self.batch_size)):
                X, Y = [], []
                for ID in batch:
                    x, y = self.make_sample(ID)
                    X.append(x)
                    Y.append(y)

                yield np.array(X) / 255, np.array(Y)
