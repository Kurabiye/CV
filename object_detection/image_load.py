import tensorflow as tf
import matplotlib.pyplot as plt
import os
import albumentations as alb
import cv2
import json
import numpy as np


#load image into tf data pipeline
def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

load_image_pipeline = False
if load_image_pipeline:
    images = tf.data.Dataset.list_files('potato/images/*.jpg')
    images.as_numpy_iterator().next()
    #print(images.as_numpy_iterator().next()) #load 1 image randomly

    images = images.map(load_image)
    images.as_numpy_iterator().next()
    #print(images.as_numpy_iterator().next()) #load image as numpy array


    #view raw images with matplotlib
    image_generator = images.batch(4).as_numpy_iterator()
    plot_images = image_generator.next()
    #print(plot_images) #load 4 images

    plot = False
    if plot:
        plt.figure(figsize=(7,7))
        for idx, image in enumerate(plot_images):
            plt.subplot(2,2,idx+1)
            plt.imshow(image)
            plt.axis("off")
        plt.show()


#apply image augmentation on images and labels using albumentations
def extract_coordinates(label):
    coords = [0,0,0,0,'']
    coord = []
    for idx in range(len(label['shapes'])):
    #extract coordinates and resclae to match image resolution
        #print("\nraw points: ",label['shapes'][idx]['points'])
        coords[0] = label['shapes'][idx]['points'][0][0]
        coords[1] = label['shapes'][idx]['points'][0][1]
        coords[2] = label['shapes'][idx]['points'][1][0]
        coords[3] = label['shapes'][idx]['points'][1][1]
        coords[4] = 'potato'

        if coords[0] > coords[2]:
            coords[0] = label['shapes'][idx]['points'][1][0]
            coords[2] = label['shapes'][idx]['points'][0][0]
        if coords[1] > coords[3]:
            coords[1] = label['shapes'][idx]['points'][1][1]
            coords[3] = label['shapes'][idx]['points'][0][1]

        coords[2] = coords[2]-coords[0]
        coords[3] = coords[3]-coords[1]
        #coords = list(np.divide(coords,[224,224,224,224]))

        coord.append([coords[0],coords[1],coords[2],coords[3],coords[4]])
        #print(f"extracted {idx} coords: ",coords)

        #print("coords array: ", coord)
        #print(idx,coord[idx])
        #print(coord)
        #print(coords)
    return coord

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

"""def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    #Visualizes a single bounding box on the image
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img"""

"""def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)"""

augmentor = alb.Compose([alb.RandomCrop(width=128,height=128),
                            alb.HorizontalFlip(p=0.5),
                            alb.RandomBrightnessContrast(p=0.2),
                            alb.RandomGamma(p=0.2),
                            alb.RGBShift(p=0.2),
                            alb.VerticalFlip(p=0.5)],
                            bbox_params=alb.BboxParams(format='coco',min_visibility=0.3))

bbox_show = False
if bbox_show:
    #load a test image and annotation with opencv and json
    img = cv2.imread(os.path.join('potato','train','images','1604f2fc-3360-11ed-bfa8-6e8209c6ae4b.jpg'))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    with open(os.path.join('potato','train','labels','1604f2fc-3360-11ed-bfa8-6e8209c6ae4b.json'),'r') as f:
        label = json.load(f)
    #print(label['shapes'][0]['points']) #print label coordinates
    #print(len(label['shapes']))
    #print(label['shapes'][0]['points'])
    
    plt.figure(figsize=(7,7))
    coord = extract_coordinates(label)
    augmented = augmentor(image=img,bboxes=coord)
    #print("\naugmented bboxes: ",augmented['bboxes'][0])
    #print(len(augmented['bboxes']))

    for idx,bbox in enumerate(augmented['bboxes']):
        cv2.rectangle(augmented['image'],(int(bbox[0]),int(bbox[1])),
                                        (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),(0,225,0),1)
    plt.imshow(augmented['image'])
    plt.axis('off')
    plt.show()

    """for idx,coords in enumerate(coord):
        try:
            #print(coords)
            coords = list(np.divide(coords,[224,224,224,224]))
            #apply augmentations and view results
            augmented = augmentor(image=img,bboxes=[coords],class_labels=['potato'])
            cv2.rectangle(augmented['image'],
                        tuple(np.multiply(augmented['bboxes'][0][:2],[124,124]).astype(int)),
                        tuple(np.multiply(augmented['bboxes'][0][2:],[124,124]).astype(int)),
                        (0,255,0),2)
            plt.subplot(3,int((len(coord))/3),idx+1)
            plt.axis("off")
            plt.imshow(augmented['image'])
            #print(np.multiply(augmented['bboxes'][0][:2],[224,224]).astype(int),np.multiply(augmented['bboxes'][0][2:],[224,224]).astype(int))
        except Exception as e:
            print(e)
    plt.show()"""


#build and run augmentation pipeline
augmentation_pipeline = False
if augmentation_pipeline:
    for partition in ['train','test','val']:
        for idx,image in enumerate(i for i in os.listdir(os.path.join('potato',partition,'images')) if i.endswith(".jpg")):
            img = cv2.imread(os.path.join('potato',partition,'images',image))
            print(idx,"image loaded: ",image)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            coords = [0,0,0.00000001,0.0000001]
            label_path = os.path.join('potato',partition,'labels',f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):
                with open(label_path,'r') as f:
                    label = json.load(f)
                    #print("label load")
                
                coords = extract_coordinates(label)
            
            try:
                for x in range(60):
                    augmented = augmentor(image=img,bboxes=coords)
                    cv2.imwrite(os.path.join('aug_data',partition,'images',f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])
                    #print(x,"augmented saved")

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0,0,0,0]
                            annotation['class'] = 0
                        else:
                            bboxes = []
                            for bbox in augmented['bboxes']:
                                bboxes.append([bbox[0],bbox[1],bbox[2],bbox[3]])
                            annotation['bbox'] = bboxes
                            annotation['class'] = 1
                            #print("annotation appended")
                    else:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0

                    with open(os.path.join('aug_data',partition,'labels',f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                        json.dump(annotation,f)
                        #print("annotation saved")
            
            except Exception as e:
                print(e)


#load augmented images to tensorflow dataset
train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg',shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x,(128,128)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg',shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x,(128,128)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg',shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x,(128,128)))
val_images = val_images.map(lambda x: x/255)

plot_tf = False
if plot_tf:
    image = train_images.as_numpy_iterator().next()
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()


#prepare labels
def load_labels(label_path):
    with open (label_path.numpy(), 'r',encoding="utf-8") as f:
        label = json.load(f)
        return [label['class']], label['bbox']

train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json',shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))
#train_sample = train_labels.as_numpy_iterator().next()
#print(train_sample)


test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json',shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json',shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x],[tf.uint8,tf.float16]))


#combine image and label samples
#print(len(train_images),len(train_labels),len(test_images),len(test_labels),len(val_images),len(val_labels)) #check partition lenghts
train = tf.data.Dataset.zip((train_images,train_labels))
#print(list(train.as_numpy_iterator()))
train = train.shuffle(5000)
#train = train.unbatch()
train = train.batch(1)
train = train.prefetch(tf.data.AUTOTUNE)

test = tf.data.Dataset.zip((test_images,test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(tf.data.AUTOTUNE)

val = tf.data.Dataset.zip((val_images,val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(tf.data.AUTOTUNE)

#print(train.as_numpy_iterator().next()[1])


#view images and annotations
data_samples = train.as_numpy_iterator()
res = data_samples.next()
print(list(res[1]))

plt.figure(figsize=(12,6))

"""for idx in range(3):
    plt.subplot(2,2,idx+1)
    sample_image = res[0][0]
    sample_coords = res[1][1][0]

    cv2.rectangle(sample_image,(int(sample_coords[0]),int(sample_coords[1])),
                (int(sample_coords[0]+sample_coords[2]),int(sample_coords[1]+sample_coords[3])),
                (0,225,0),1)
    
    plt.imshow(sample_image)

plt.show()"""


