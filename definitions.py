# Train and validation directory
TRAIN_BATCH_SIZE = 5
VAL_BATCH_SIZE = 5

train_image_folder = 'imgs/train'
train_annot_folder = 'annotations/train'
val_image_folder = 'imgs/validation'
val_annot_folder = 'annotations/validation'

LABELS           = ('class')
IMAGE_H, IMAGE_W = 512, 512
GRID_H,  GRID_W  = 16, 16 # GRID size = IMAGE size / 32
BOXES_CELL       = 5
NUM_CLASSES      = len(LABELS)
SCORE_THRESHOLD  = 0.5
IOU_THRESHOLD    = 0.45
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

TRAIN_BATCH_SIZE = 10
VAL_BATCH_SIZE   = 10
EPOCHS           = 3

LAMBDA_NOOBJECT  = 1
LAMBDA_OBJECT    = 5
LAMBDA_CLASS     = 1
LAMBDA_COORD     = 1