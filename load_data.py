import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa

from definitions import *

# Parses the annotations from the .txt files into a numpy array
def parse_annotation(ann_dir, img_dir):
    '''
    Parse txt files with annotations.
    
    Parameters
    ----------
    - ann_dir : annotations files directory
    - img_dir : images files directory
    
    Returns
    -------
    - imgs_name : numpy array of images files path (shape : images count, 1)
    - true_boxes : numpy array of annotations for each image (shape : image count, max annotation count, 5)
        annotation format :  num_vertexes, x_bl, x_tl, x_tr, x_br,
            y_bl, y_tl, y_tr, y_br, class_name
        class = 'class' (there is only one for now)
    '''
    
    max_annot = 0
    imgs_name = []
    annots = []

    
    # Iterate over directory and parse files
    for ann_name in sorted(os.listdir(ann_dir)):
        # Check if image exists
        img_name = os.path.splitext(ann_name)[0] + ".jpg"
        img = ""
        try:
            img = open(os.path.join(img_dir, img_name))
        except:
            # Stop the function if it doesn't
            print(f"Error: Image {img_name} not found!")
            return
        imgs_name.append(os.path.join(img_dir, img_name))

        #Start parsing the annotations
        ann = open(os.path.join(ann_dir, ann_name))
        annot_count = 0
        boxes = []
        line = ann.readline()
        # Read all lines in file
        while line:
            # Read and convert info (vertex count and class name are ignored)         

            x_bl, x_tl, x_tr, x_br, y_bl, y_tl, y_tr, y_br, = [float(coord) for coord in line.split(",")[1:-2]]
            box = [x_bl, y_bl,
                   x_tl, y_tl,
                   x_tr, y_tr,
                   x_br, y_br]
            boxes.append(np.asarray(box))

            # Count the number of annotations in file
            annot_count += 1

            line = ann.readline()

        annots.append(np.asarray(boxes))

        if annot_count > max_annot:
                max_annot = annot_count

        ann.close()
        img.close()
      
    # Rectify annotations boxes : len -> max_annot
    imgs_name = np.array(imgs_name)
    true_boxes = np.zeros((imgs_name.shape[0], max_annot, 8))
    for idx, boxes in enumerate(annots):
        true_boxes[idx, :boxes.shape[0], :8] = boxes

    return imgs_name, true_boxes

# Parses the annotations from the .txt files into a numpy array (using horizontal bounding boxes)
def parse_annotation_hbb(ann_dir, img_dir):
    '''
    Parse txt files with annotations.
    Using horizontal bounding boxes.
    
    Parameters
    ----------
    - ann_dir : annotations files directory
    - img_dir : images files directory
    
    Returns
    -------
    - imgs_name : numpy array of images files path (shape : images count, 1)
    - true_boxes : numpy array of annotations for each image (shape : image count, max annotation count, 5)
        annotation format :  num_vertexes, x_bl, x_tl, x_tr, x_br,
            y_bl, y_tl, y_tr, y_br, class_name
        class = 'class' (there is only one for now)
    '''
    
    max_annot = 0
    imgs_name = []
    annots = []

    
    # Iterate over directory and parse files
    for ann_name in sorted(os.listdir(ann_dir)):
        # Check if image exists
        img_name = os.path.splitext(ann_name)[0] + ".jpg"
        img = ""
        try:
            img = open(os.path.join(img_dir, img_name))
        except:
            # Stop the function if it doesn't
            print(f"Error: Image {img_name} not found!")
            return
        imgs_name.append(os.path.join(img_dir, img_name))

        #Start parsing the annotations
        ann = open(os.path.join(ann_dir, ann_name))
        annot_count = 0
        boxes = []
        line = ann.readline()
        # Read all lines in file
        while line:
            # Read and convert info (vertex count and class name are ignored)         

            x_bl, x_tl, x_tr, x_br, y_bl, y_tl, y_tr, y_br, = [float(coord) for coord in line.split(",")[1:-2]]

            x_min = min(x_bl, x_tl)
            x_max = max(x_br, x_tr)

            y_min = min(y_bl, y_br)
            y_max = max(y_tl, y_tr)

            # Make a box that goes from the leftmost-topmost point to the rightmost-topmost point (1 for class, 0 is no class)
            box = [x_min, y_max, x_max, y_min, 1]
            # Create the box ()
            boxes.append(np.asarray(box))

            # Count the number of annotations in file
            annot_count += 1

            line = ann.readline()

        annots.append(np.asarray(boxes))

        if annot_count > max_annot:
                max_annot = annot_count

        ann.close()
        img.close()
      
    # Rectify annotations boxes : len -> max_annot
    imgs_name = np.array(imgs_name)
    true_boxes = np.zeros((imgs_name.shape[0], max_annot, 5))
    for idx, boxes in enumerate(annots):
        true_boxes[idx, :boxes.shape[0], :5] = boxes

    return imgs_name, true_boxes

# Parses the images into tensorflow objects
def parse_function(img_obj, true_boxes):
    x_img_string = tf.io.read_file(img_obj)
    x_img = tf.image.decode_png(x_img_string, channels=3) # dtype=tf.uint8
    x_img = tf.image.convert_image_dtype(x_img, tf.float32) # pixel value /255, dtype=tf.float32, channels : RGB
    x_img = tf.image.resize(x_img, (IMAGE_W, IMAGE_W)) # convert all images to yolo grid style
    return x_img, true_boxes

# Actually "generates" the dataset using previous functions
def get_dataset(img_dir, ann_dir, batch_size):
    '''
    Create a YOLO dataset

    Parameters
    ----------
    - ann_dir : annotations files directory
    - img_dir : images files directory
    - labels : labels list
    - batch_size : int

    Returns
    -------
    - YOLO dataset : generate batch
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    Note : image pixel values = pixels value / 255. channels : RGB
    '''
    imgs_name, bbox = parse_annotation_hbb(ann_dir, img_dir)
    dataset = tf.data.Dataset.from_tensor_slices((imgs_name, bbox))
    dataset = dataset.shuffle(len(imgs_name))
    dataset = dataset.repeat()
    dataset = dataset.map(parse_function, num_parallel_calls=6)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    print('-------------------')
    print('Dataset:')
    print('Images count: {}'.format(len(imgs_name)))
    print('Step per epoch: {}'.format(len(imgs_name) // batch_size))
    print('Images per epoch: {}'.format(batch_size * (len(imgs_name) // batch_size)))
    return dataset

# Test dataset
def test_dataset(dataset):
    batch_idx = 0
    for batch in dataset:
        img = batch[0][0]
        label = batch[1][0]
        plt.figure(figsize=(2, 2))
        f, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
        ax1.imshow(img)
        ax1.set_title('Input image. Shape : {}'.format(img.shape))
        for i in range(label.shape[0]):
            box = label[i, :]
            x = box[0] * img.shape[1]
            y = box[3] * img.shape[0]
            w = (box[2] - box[0]) * img.shape[1]
            h = (box[1] - box[3]) * img.shape[0]
            color = (0, 1, 0)
            rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
        plt.savefig(f'images_test/image{batch_idx}.png')
        batch_idx += 1
        if batch_idx == 10:
            break

def augmentation_generator(yolo_dataset):
    '''
    Augmented batch generator from a yolo dataset

    Parameters
    ----------
    - YOLO dataset
    
    Returns
    -------
    - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch : tuple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    '''
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1]. numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        for i in range(img.shape[0]):
            ia_bbs = [ia.BoundingBox(x1=bb[0],
                                     y1=bb[1],
                                     x2=bb[2],
                                     y2=bb[3]) for bb in boxes[i]
                      if (bb[0] + bb[1] +bb[2] + bb[3] > 0)]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(1, 1)))
        # data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.2, 1.2)), # change brightness
            ])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        for i in range(img.shape[0]):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i,j,0] = bb.x1
                boxes[i,j,1] = bb.y1
                boxes[i,j,2] = bb.x2
                boxes[i,j,3] = bb.y2
        # conversion numpy->tensor
        batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        yield batch

def process_true_boxes(true_boxes, anchors, image_width, image_height):
    '''
    Build image ground truth in YOLO format from image true_boxes and anchors.
    
    Parameters
    ----------
    - true_boxes : tensor, shape (max_annot, 5), format : x1 y1 x2 y2 c, coords unit : image pixel
    - anchors : list [anchor_1_width, anchor_1_height, anchor_2_width, anchor_2_height...]
        anchors coords unit : grid cell
    - image_width, image_height : int (pixels)
    
    Returns
    -------
    - detector_mask : array, shape (GRID_W, GRID_H, anchors_count, 1)
        1 if bounding box detected by grid cell, else 0
    - matching_true_boxes : array, shape (GRID_W, GRID_H, anchors_count, 5)
        Contains adjusted coords of bounding box in YOLO format
    -true_boxes_grid : array, same shape than true_boxes (max_annot, 5),
        format : x, y, w, h, c, coords unit : grid cell
        
    Note:
    -----
    Bounding box in YOLO Format : x, y, w, h, c
    x, y : center of bounding box, unit : grid cell
    w, h : width and height of bounding box, unit : grid cell
    c : label index
    ''' 
    
    scale = IMAGE_W / GRID_W # scale = 32
    
    anchors_count = len(anchors) // 2
    anchors = np.array(anchors)
    anchors = anchors.reshape(len(anchors) // 2, 2)
    
    detector_mask = np.zeros((GRID_W, GRID_H, anchors_count, 1))
    matching_true_boxes = np.zeros((GRID_W, GRID_H, anchors_count, 5))
    
    # convert true_boxes numpy array -> tensor
    true_boxes = true_boxes.numpy()
    
    true_boxes_grid = np.zeros(true_boxes.shape)
    
    # convert bounding box coords and localize bounding box
    for i, box in enumerate(true_boxes):
        # convert box coords to x, y, w, h and convert to grids coord
        w = (box[2] - box[0]) / scale
        h = (box[3] - box[1]) / scale    
        x = ((box[0] + box[2]) / 2) / scale
        y = ((box[1] + box[3]) / 2) / scale
        true_boxes_grid[i,...] = np.array([x, y, w, h, box[4]])
        if w * h > 0: # box exists
            # calculate iou between box and each anchors and find best anchors
            best_iou = 0
            best_anchor = 0
            for i in range(anchors_count): 
                # iou (anchor and box are shifted to 0,0)
                intersect = np.minimum(w, anchors[i,0]) * np.minimum(h, anchors[i,1])
                union = (anchors[i,0] * anchors[i,1]) + (w * h) - intersect
                iou = intersect / union
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            # localize box in detector_mask and matching true_boxes
            if best_iou > 0:
                x_coord = np.floor(x).astype('int')
                y_coord = np.floor(y).astype('int')
                detector_mask[y_coord, x_coord, best_anchor] = 1
                yolo_box = np.array([x, y, w, h, box[4]])
                matching_true_boxes[y_coord, x_coord, best_anchor] = yolo_box
    return matching_true_boxes, detector_mask, true_boxes_grid

def ground_truth_generator(dataset):
    '''
    Ground truth batch generator from a yolo dataset, ready to compare with YOLO prediction in loss function.

    Parameters
    ----------
    - YOLO dataset. Generate batch:
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
        
    Returns
    -------
    - imgs : images to predict. tensor (shape : batch_size, IMAGE_H, IMAGE_W, 3)
    - detector_mask : tensor, shape (batch, size, GRID_W, GRID_H, anchors_count, 1)
        1 if bounding box detected by grid cell, else 0
    - matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
        Contains adjusted coords of bounding box in YOLO format
    - class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
        One hot representation of bounding box label
    - true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
        true_boxes format : x, y, w, h, c, coords unit : grid cell
    '''
    for batch in dataset:
        # imgs
        imgs = batch[0]
        
        # true boxes
        true_boxes = batch[1]
        
        # matching_true_boxes and detector_mask
        batch_matching_true_boxes = []
        batch_detector_mask = []
        batch_true_boxes_grid = []
        
        for i in range(true_boxes.shape[0]):     
            one_matching_true_boxes, one_detector_mask, true_boxes_grid = process_true_boxes(true_boxes[i],
                                                                                           ANCHORS,
																						   imgs[i].shape[1],
                                                                                           imgs[i].shape[0],
                                                                                           )
            batch_matching_true_boxes.append(one_matching_true_boxes)
            batch_detector_mask.append(one_detector_mask)
            batch_true_boxes_grid.append(true_boxes_grid)
                
        detector_mask = tf.convert_to_tensor(np.array(batch_detector_mask), dtype='float32')
        matching_true_boxes = tf.convert_to_tensor(np.array(batch_matching_true_boxes), dtype='float32')
        true_boxes_grid = tf.convert_to_tensor(np.array(batch_true_boxes_grid), dtype='float32')
        
        # class one_hot
        matching_classes = K.cast(matching_true_boxes[..., 4], 'int32') 
        class_one_hot = K.one_hot(matching_classes, CLASS + 1)[:,:,:,:,1:]
        class_one_hot = tf.cast(class_one_hot, dtype='float32')
        
        batch = (imgs, detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid)
        yield batch
