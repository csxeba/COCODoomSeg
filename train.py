import data
from architecture import fcnn


dataset = data.COCODoomDataset("/data/Datasets/cocodoom", "val")
model = fcnn.build(dataset.num_classes)
model.summary()
model.fit_generator(dataset.stream(batch_size=16), steps_per_epoch=len(dataset.index) // 16, epochs=30)
