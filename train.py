import datetime
import os

from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

import data
from architecture import fcnn

NOW = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
BATCH_SIZE = 32

artifactory = "/drive/My Drive/artifactory/COCODoomSeg/"
checkpoint_to = os.path.join(artifactory, "checkpoint_{}.h5")

if not os.path.exists(artifactory):
    os.makedirs(artifactory)

train_ds = data.COCODoomDataset("/data/Datasets/cocodoom", "train", BATCH_SIZE)
val_ds = data.COCODoomDataset("/data/Datasets/cocodoom", "val", BATCH_SIZE,
                              ignored_classes=train_ds.ignores)

model = fcnn.build(train_ds.num_classes)

callbacks = [
    ModelCheckpoint(checkpoint_to.format("latest")),
    ModelCheckpoint(checkpoint_to.format("best"), save_best_only=True),
    CSVLogger(os.path.join(artifactory, "training_log.csv")),
    TensorBoard(os.path.join(artifactory, "tensorboard"), write_graph=False)
]

model.fit_generator(train_ds.stream(shuffle=True),
                    steps_per_epoch=train_ds.steps_per_epoch,
                    epochs=120,
                    validation_data=val_ds.stream(shuffle=False),
                    validation_steps=val_ds.steps_per_epoch,
                    callbacks=callbacks)
