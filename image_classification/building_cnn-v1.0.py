import tensorflow as tf
from keras import layers,models,utils,Sequential,losses,callbacks
import matplotlib.pyplot as plt
import numpy as np
import cv2


image_size = 256
channels = 3
batch_size = 32
epochs = 30

dataset = utils.image_dataset_from_directory(
    "dataset/potato_leaf",
    shuffle=True,
    image_size=(image_size,image_size),
    batch_size=batch_size
)

class_names = dataset.class_names
print(class_names)


#plt.figure(figsize=(5,5))

#plot images random
"""for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.axis("off")
        plt.title(class_names[label_batch[i]])
plt.show()"""

#getting dataset partition
"""train_size = 0.8
val_size = 0.1
test_size = 0.1
train_ds = dataset.take(int(len(dataset)*train_size))
test_ds = dataset.skip(len(train_ds))
val_ds = test_ds.take(int(len(dataset)*val_size))
test_ds = test_ds.skip(len(val_ds))"""

#print(f"train size: {len(train_ds)}, val size: {len(val_ds)}, test size: {len(test_ds)}")

def get_dataset_partition_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000):

    if shuffle:
        ds = ds.shuffle(shuffle_size,seed=12)

    ds_size = len(ds)
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)

    train_ds = ds.take(train_size)

    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)


    return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds = get_dataset_partition_tf(dataset)
#print(f"train size: {len(train_ds)}, val size: {len(val_ds)}, test size: {len(test_ds)}")

traind_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = Sequential([
    layers.Resizing(image_size,image_size),
    layers.Rescaling(1.0/255)
])

data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

model = models.Sequential([
    resize_and_rescale,
    data_augmentation
])

input_shape = (batch_size,image_size,image_size,channels)
n_classes = 3

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(n_classes,activation='softmax'))

model.build(input_shape=input_shape)
#model.summary()

model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
checkpoint_path = "model/potato_leaf/weights.hdf5"
callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=True,
                                verbose=1,monitor='accuracy',mode='auto',period=1)

training = False

if training:
    history = model.fit(
        train_ds,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=val_ds,
        callbacks=callback
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    history_arr = np.array([acc,val_acc,loss,val_loss])
    np.save("model/potato_leaf/history.npy",history_arr)

else:
    model.load_weights(checkpoint_path)
    history_arr = np.load("model/potato_leaf/history.npy")
    acc = history_arr[0]
    val_acc = history_arr[1]
    loss = history_arr[2]
    val_loss = history_arr[3]

plot = False
if plot:
    #Accuracy
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(range(epochs),acc, label='Training Accuracy')
    plt.plot(range(epochs),val_acc,label='Validation Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.legend(loc='lower right')
    #Loss
    plt.subplot(1,2,2)
    plt.plot(range(epochs),loss,label='Training Loss')
    plt.plot(range(epochs),val_loss,label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.show()

def prediction(model,img):
    img_array = utils.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    #print("prediction 0: ",(predictions[0]*10))
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence

predict = True
if predict:
    """plt.figure(figsize=(12,7))
    for images, labels in test_ds.take(1):
        for i in range(8):
            #print("actual label: ",class_names[first_label])
            #plt.text(10,10,f"actual label{class_names[first_label]}",fontsize=10,color='r')
            #print("predicted label: ",class_names[np.argmax(batch_prediction[0])])
            #plt.text(10,30,f"predicted label: {class_names[np.argmax(batch_prediction[0])]}",fontsize=10,color='r')
            #print("first image to predict")

            ax = plt.subplot(2,4,i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class,confidence = prediction(model,images[i].numpy())
            actual_class = class_names[labels[i]]
            print(f"Actual: {actual_class}, Predicted: {predicted_class} Confidence: %{confidence}")
            
            plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}\n Confidence: %{confidence}",fontsize=10)
            plt.axis("off")

        plt.show()"""
    
    test_model1 = cv2.imread("dataset/potato/Test/1003.jpg")
    test_model2 = cv2.imread("dataset/potato_leaf/Potato___Late_blight/3f7f719f-9849-47c5-8f79-0384a64f8e8f___RS_LB 2862.JPG")
    test_model_arr = [test_model1,test_model2]
    for i in range(2):
        test_model = cv2.resize(test_model_arr[i],(image_size,image_size),interpolation=cv2.INTER_AREA)
        test_model = test_model.reshape(1,image_size,image_size,3)
        prediction = model.predict(test_model)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = (prediction[0])
        print(predicted_class,"confidence: ",confidence)
        confidence = ((prediction[0]))
        print(confidence)
