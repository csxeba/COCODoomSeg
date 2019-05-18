import os
import json
from collections import defaultdict

import numpy as np
import cv2

import masking


class COCODoomDataset:

    BACKGROUND_CLASS = 0

    def __init__(self,
                 data,
                 image_root,
                 version="standard",
                 batch_size=16,
                 ignored_class_ids=None):

        self.root = image_root
        self.batch_size = batch_size
        self.ignored_class_ids = ignored_class_ids or set()

        version_str = {"standard": "", "full": "-full"}[version]

        if not isinstance(data, dict):
            data = json.load(open(data))

        self.index = defaultdict(list)
        for anno in data["annotations"]:
            self.index[anno["image_id"]].append(anno)
        self.image_meta = {meta["id"]: meta for meta in data["images"]}
        self.classes = {}
        self.num_classes = 1

        for category in data["categories"]:
            ID = category["id"]
            if ID in self.ignored_class_ids:
                self.classes[ID] = 0
            else:
                self.classes[ID] = self.num_classes
                self.num_classes += 1

        print(f"Num images :", len(data["images"]))
        print(f"Num annos  :", len(data["annotations"]))
        print(f"Num classes:", self.num_classes)

    @property
    def steps_per_epoch(self):
        return len(self.index) // self.batch_size

    def _mask_sparse(self, image_shape, image_id):
        mask = np.zeros(image_shape[:2] + (self.num_classes+1,))
        for anno in self.index[image_id]:
            instance_mask = masking.get_mask(anno, image_shape[:2])
            category = self.classes[anno["category_id"]]
            if category == 0:
                continue
            mask[..., category][instance_mask] = 1

        overlaps = mask.sum(axis=2)[..., None]
        overlaps[overlaps == 0] = 1
        mask /= overlaps
        mask[..., 0] = 1 - mask[..., 1:].sum(axis=2)
        return mask

    def _mask_dense(self, image_shape, image_id):
        mask = np.zeros(image_shape[:2])
        for anno in self.index[image_id]:
            instance_mask = masking.get_mask(anno, image_shape[:2])
            category = self.classes[anno["category_id"]]
            if category == 0:
                continue
            mask[instance_mask] = category
        return mask[..., None]

    def make_sample(self, image_id, sparse_y=True):
        meta = self.image_meta[image_id]
        image_path = os.path.join(self.root, meta["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"No image found @ {image_path}")
        if sparse_y:
            mask = self._mask_sparse(image.shape, image_id)
        else:
            mask = self._mask_dense(image.shape, image_id)
        return image, mask

    def stream(self, shuffle=True, sparse_y=True, run=None, level=None):
        meta_iterator = self.image_meta.values()
        if run is not None:
            criterion = "run{}".format(run)
            meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
        if level is not None:
            criterion = "map{:0>2}".format(level)
            meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)

        ids = sorted(meta["id"] for meta in meta_iterator)
        N = len(ids)

        while 1:
            if shuffle:
                np.random.shuffle(ids)
            for batch in (ids[start:start+self.batch_size] for start in range(0, N, self.batch_size)):
                X, Y = [], []
                for ID in batch:
                    x, y = self.make_sample(ID, sparse_y)
                    X.append(x)
                    Y.append(y)

                yield np.array(X) / 255, np.array(Y)
