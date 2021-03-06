import numpy as np
import cv2

import data


class Visualizer:

    ENEMY_TYPES = [
        "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
        "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"
    ]
    COLORS = dict(zip(ENEMY_TYPES, [
        c.RED, c.BLUE, c.RED, c.BLUE, c.WHITE, c.GREEN, c.YELLOW, c.PINK, c.RED, c.GREEN, c.GREY, c.RED,
        c.WHITE, c.WHITE, c.WHITE, c.WHITE, c.BLUE
    ]))

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.colors = np.random.randint(50, 201, size=(n_classes, 3))

    def _handle_sparse(self, y):
        segmentation = np.zeros(y.shape[:2] + (3,), dtype="uint8")
        for i in range(1, self.n_classes):
            segmentation[y[..., i] > 0.5] = self.colors[i]
        segmentation = np.ma.masked_array(segmentation, mask=segmentation == 0)
        return segmentation

    def _handle_dense(self, y):
        segmentation = np.zeros(y.shape[:2] + (3,), dtype="uint8")
        for i in range(1, self.n_classes):
            indices = np.where(y == i)
            segmentation[indices[0], indices[1]] = self.colors[i]
        segmentation = np.ma.masked_array(segmentation, mask=segmentation == 0)
        return segmentation

    def colorify_mask(self, y):
        if y.ndim == 4:
            y = y[0]
        if y.shape[-1] == 1:
            return self._handle_dense(y)
        else:
            return self._handle_sparse(y)


if __name__ == '__main__':
    dataset = data.COCODoomDataset("/data/Datasets/cocodoom", "val", batch_size=1)
    visualizer = Visualizer(dataset.num_classes)

    for [x], [y] in dataset.stream(shuffle=False, sparse_y=False):
        w = visualizer.colorify_mask(x, y)
        cv2.imshow("frame", w)
        if cv2.waitKey(1000 // 10) == 28:
            break
