import datetime
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

import data
from architecture import fcnn
import visualize

np.random.seed(1337)

NOW = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
BATCH_SIZE = 8

checkpoint_to = "/data/My Drive/artifactory/COCODoomSeg/checkpoint_{}.h5"

# train_ds = data.COCODoomDataset("/data/Datasets/cocodoom", "train", BATCH_SIZE)
val_ds = data.COCODoomDataset("/data/Datasets/cocodoom", "val", BATCH_SIZE)


def fake_stream():
    x, y = next(val_ds.stream())
    while 1:
        yield x, y


model = fcnn.build(val_ds.num_classes)

stream = fake_stream()

if not os.path.exists("model.h5"):
    model.fit_generator(stream,
                        steps_per_epoch=200,
                        epochs=1)

    model.save_weights("model.h5")
else:
    model.load_weights("model.h5")

x, y = next(stream)
pred = model.predict(x)
vis = visualize.Visualizer(val_ds.num_classes)
for xx, yy, pp in zip(x, y, pred):
    image = cv2.cvtColor((xx * 255).astype("uint8"), cv2.COLOR_BGR2RGB)

    gt = vis.overlay_mask(xx, yy)
    dt = vis.overlay_mask(xx, pp)

    fig, (l, r) = plt.subplots(1, 2, figsize=(9, 6))
    l.imshow(image)
    l.imshow(gt, alpha=0.5)

    r.imshow(image)
    r.imshow(dt, alpha=0.5)

    l.set_title("Ground Truth")
    r.set_title("Detection")
    plt.tight_layout()
    plt.show()
