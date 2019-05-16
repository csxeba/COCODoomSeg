import datetime

import data
from architecture import fcnn
import visualize

NOW = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
BATCH_SIZE = 4

checkpoint_to = "/data/My Drive/artifactory/COCODoomSeg/checkpoint_{}.h5"

# train_ds = data.COCODoomDataset("/data/Datasets/cocodoom", "train", BATCH_SIZE)
val_ds = data.COCODoomDataset("/data/Datasets/cocodoom", "val", BATCH_SIZE)

model = fcnn.build(val_ds.num_classes)
model.summary()

def fake_stream():
    x, y = next(val_ds.stream())
    while 1:
        yield x, y

stream = fake_stream()
model.fit_generator(stream,
          steps_per_epoch=200,
          epochs=1)

from matplotlib import pyplot as plt

x, y = next(stream)
pred = model.predict(x)
vis = visualize.Visualizer(val_ds.num_classes)
for xx, yy, pp in zip(x, y, pred):
    gt = vis.overlay_mask(xx, yy)
    dt = vis.overlay_mask(xx, pred)
    fig, (l, r) = plt.subplots(1, 2)
    l.imshow(gt)
    r.imshow(dt)
    l.set_title("Ground Truth")
    r.set_title("Detection")
    plt.show()
