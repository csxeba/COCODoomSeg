import numpy as np
import cv2

import data


class Visualizer:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.colors = np.random.randint(50, 201, size=(n_classes, 3))

    def overlay_mask(self, x, y):
        if x.ndim == 4:
            x = x[0]
        if y.ndim == 4:
            y = y[0]
        segmentation = np.zeros_like(x, dtype="uint8")
        for i in range(self.n_classes):
            segmentation[y[..., i] > 0] = self.colors[i]
        segmentation = np.ma.masked_array(segmentation, mask=segmentation == 0)
        weighted = cv2.addWeighted(x, 0.5, segmentation, 0.5, 1.)
        return weighted


dataset = data.COCODoomDataset("/data/Datasets/cocodoom", "val")
visualizer = Visualizer(dataset.num_classes)

for [x], [y] in dataset.stream(batch_size=1, shuffle=False):
    w = visualizer.overlay_mask(x, y, dataset.num_classes)
    cv2.imshow("frame", w)
    if cv2.waitKey(1000 // 10) == 28:
        break
