import numpy as np
import cv2

from architecture import fcnn
from data import COCODoomDataset
from visualize import Visualizer


def categorical_crossentropy(pred, y):
    losses = (-y * np.log(pred)).sum(axis=(-1))
    losses /= losses.max()
    losses *= 255
    losses = losses.astype("uint8")
    return losses


ds = COCODoomDataset("/data/Datasets/cocodoom", "val", batch_size=1)
net = fcnn.build(ds.num_classes)
net.load_weights("checkpoint_best.h5")

vis = Visualizer(ds.num_classes)

for X, [y] in ds.stream(shuffle=False):
    pred = net.predict(X)[0]

    gt = vis.overlay_mask(X[0], y)
    dt = vis.overlay_mask(X[0], y)

    x = cv2.cvtColor((X[0]*255).astype("uint8"), cv2.COLOR_BGR2RGBA)
    x[..., -1] = y[0]

    cv2.imshow("Image", x)
    cv2.imshow("GroundTruth", gt)
    cv2.imshow("Detection", dt)
    cv2.imshow("Categorical Crossentropy", categorical_crossentropy(pred, y))
    cv2.waitKey(1000//5)
