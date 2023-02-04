from definitions import *
from load_data import *
from losses import *
from train import *
import MODEL_yolov2

print('Tensorflow version : {}'.format(tf.__version__))
print('GPU : {}'.format(tf.config.list_physical_devices('GPU')))

train_dataset = None
train_dataset= get_dataset(train_image_folder, train_annot_folder, TRAIN_BATCH_SIZE)

val_dataset = None
val_dataset= get_dataset(val_image_folder, val_annot_folder, VAL_BATCH_SIZE)

aug_train_dataset = augmentation_generator(train_dataset)
test_dataset(aug_train_dataset)

train_gen = ground_truth_generator(aug_train_dataset)
val_gen = ground_truth_generator(val_dataset)

model = MODEL_yolov2.yolov2(IMAGE_H, IMAGE_W, BOXES_CELL, NUM_CLASSES)

results = train(EPOCHS, model, train_gen, val_gen, 10, 2, 'training_2')

test_loss(train_gen, model, hbb_yolov2_loss)

results = train(EPOCHS, model, train_gen, val_gen, 10, 2, 'training_2')

plt.plot(results[0])
plt.plot(results[1])

