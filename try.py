import os
import json

import numpy as np
import cv2
import imageio

from architecture import fcnn
from data import COCODoomDataset
from visualize import Visualizer
from utils.data_utils import ignore_categories


def sparse_categorical_crossentropy(pred, y, num_classes):
    losses = np.zeros(pred.shape[:2])
    for i in range(num_classes+1):
        mask = y[..., 0] == i
        losses[mask] = -np.log(pred[..., i][mask])

    losses /= losses.max()
    losses = np.stack([losses]*3, axis=-1)
    return losses


def overlay(x, y):
    mask = y > 0
    x[mask] = 0.5 * x[mask] + 0.5 * y[mask]
    return x


def set_title(x, title):
    return cv2.putText(x, title, (20, 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)


ROOT = "/data/Datasets/cocodoom/"
TRAIN = os.path.join(ROOT, "run-train.json")
JSON = os.path.join(ROOT, "run-full-train.json")
LVL = None
FPS = 24

train_data = json.load(open(JSON))
ignores = ignore_categories(train_data)

ds = COCODoomDataset(JSON, ROOT, batch_size=1, ignored_class_ids=ignores)
net = fcnn.build(ds.num_classes)
net.load_weights("checkpoint_best.h5")

vis = Visualizer(ds.num_classes)
writer = imageio.get_writer("COCODoomOutput_{}FPS_LVL{}.mp4".format(FPS, LVL), fps=FPS)

for i, (X, [y]) in enumerate(ds.stream(shuffle=False, level=LVL, sparse_y=False)):
    image_restored = np.clip(X[0] * 255 + 50, 0, 255).astype("uint8")

    pred = net.predict(X)[0]
    delta = sparse_categorical_crossentropy(pred, y, ds.num_classes)

    gt = vis.colorify_mask(y)
    dt = vis.colorify_mask(pred)

    gt = set_title(gt, "Label")
    overlayed = overlay(image_restored.copy(), dt)
    overlayed = set_title(overlayed, "Detection overlayed")
    delta = set_title((delta*255).astype("uint8"), "Detection error")
    image_restored = set_title(image_restored, "Image")

    top = np.concatenate([image_restored, gt], axis=1)
    bot = np.concatenate([overlayed, delta], axis=1)
    display = np.concatenate([top, bot], axis=0)
    display = cv2.resize(display, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    writer.append_data(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

    cv2.imshow("Image", display)
    cv2.waitKey(1)

    if i >= 4200:
        break

print(" -- THE END -- ")
