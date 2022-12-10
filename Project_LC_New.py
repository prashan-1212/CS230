#!/usr/bin/env python
# coding: utf-8

# ## ***Data preparation ***:

# In[ ]:


#upload the dataset and unzip
#!ls -lah '/content/new_new.zip'
from zipfile import ZipFile
file_name = "new_new.zip"
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('Done')


# In[ ]:


get_ipython().system('pip install patchify')
get_ipython().system('pip install numpy')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install Pillow')


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


import os 
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import random


# In[ ]:


minmaxscaler = MinMaxScaler()


# In[ ]:


dataset_root_folder = '/home/ubuntu' #'/home/ubuntu'#
dataset_name = "new_new" #"train"


# In[ ]:


#learn the images in the folders
for path, subdirs, files in os.walk(dataset_root_folder):
    for name in subdirs: # printing folder names
        dir_name = os.path.join(path, name)
        #print(dir_name)
    for name in files:    #printing all file names
        dir_name = (os.path.join(path, name))
        #print(dir_name)
    dir_name = path.split(os.path.sep)[-1]
    #print(dir_name)       #printing all the folder names without the path
    if dir_name == 'images':  # selecting the folders to go through files
        images = os.listdir(path) #creating a list of images
        #print(images)
        for i, image_name in enumerate(images): #enumerate through the list of images
            #print(image_name)
            if(image_name.endswith('.jpg')):    #print the jpegs 20min
                #print(image_name)
                P = True



# In[ ]:


# image = cv2.imread('DubaiDataset\Tile 1\images\image_part_001.jpg',1)
image = cv2.imread('/home/ubuntu/new_new/images/119_sat.jpg',1) #/content/new_new/images/119_sat.jpg
# print(image)
#for AWS
print(image.shape) #shape
print(type(image))  #type of th image variable 


# In[ ]:


#defining image patch size 
image_patch_size = 256
(image.shape[0]//image_patch_size)*image_patch_size


# In[ ]:


image_patches = patchify(image,(image_patch_size, image_patch_size,3),step =image_patch_size )
len(image_patches)


# In[ ]:


#reading image 
image_dataset = []
mask_dataset = []

for image_type in ['images', 'masks']:
  if image_type == 'images':
    image_extension = 'jpg'
    image_ann  = 'sat'
  elif image_type == 'masks':
    image_extension = 'png'
    image_ann  = 'mask'
  #image_type = 'images'  # images or masks
  #image_extension ='jpg'  # jpg or png
  for image_id in range(1,15000):
    #       image = cv2.imread(f'DubaiDataset\Tile {tile_id}\{image_type}\image_part_00{image_id}.{image_extension}', 1)
        image = cv2.imread(f'/home/ubuntu/new_new/{image_type}/{image_id}_{image_ann}.{image_extension}', 1)
    #         print(image)
        if image is not None:
            if image_type == 'masks':
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              #print(image.shape)
            size_x = (image.shape[1]//image_patch_size)*image_patch_size  #adjusting image size if needed
            size_y = (image.shape[0]//image_patch_size)*image_patch_size  #adjusting image size if needed
            #print("{} --- {} - {}".format(image.shape, size_y, size_x))
            image = Image.fromarray(image)  # converting to pillow image type
            image = image.crop((0,0, size_x, size_y)) #croping it to 256 multiples 
            #print("({}, {})".format(image.size[0], image.size[1]))
            image = np.array(image) # converting the image back to np array to use with patchify 
            patched_images = patchify(image,(image_patch_size, image_patch_size,3),step =image_patch_size ) # creating patches of images 
            #print(len(patched_images))
            for i in range(patched_images.shape[0]):
              for j in range(patched_images.shape[1]): 
                if image_type == 'images':
                  individual_patched_image = patched_images[i,j,:,:]
                  #print(individual_patched_image.shape)
                  individual_patched_image  = minmaxscaler.fit_transform(individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(individual_patched_image.shape)
                  individual_patched_image  = individual_patched_image[0]
                  #print(individual_patched_image.shape)
                  image_dataset.append(individual_patched_image)
                if image_type == 'masks':
                  individual_patched_mask = patched_images[i,j,:,:]
                  individual_patched_mask = individual_patched_mask[0]
                  mask_dataset.append(individual_patched_mask)


# In[ ]:


print(len(image_dataset))
print(len(mask_dataset))


# In[ ]:


#creating a numpy array from both datasets
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)


# In[ ]:


print(len(image_dataset))
print(len(mask_dataset))


# In[ ]:


#checking the images
random_image_id = random.randint(0, len(image_dataset))  # viewing random images from the dataset 

#plotting the images 
plt.figure(figsize = (14,8))
plt.subplot(121)
plt.imshow(image_dataset[random_image_id])
plt.subplot(122)
plt.imshow(mask_dataset[random_image_id])


# In[ ]:


#convert hex to RGB for the classes

class_urban_land = [ 0, 255, 255]

class_urban_land = np.array([ 0, 255, 255])
print(class_urban_land)

class_agriculture_land = [ 255,255,0]

class_agriculture_land = np.array([ 255,255,0])
print(class_agriculture_land )

class_rangeland = [ 255,0,255]

class_rangeland = np.array([ 255,0,255])
print(class_rangeland )

class_forest_land = [ 0,255,0]

class_forest_land = np.array([ 0,255,0])
print(class_forest_land  )

class_water= [0,0,255]

class_water = np.array([0,0,255])
print(class_water )

class_barren_land= [ 255,255,255]

class_barren_land = np.array([ 255,255,255])
print(class_barren_land )

class_unknown= [ 0,0,0]

class_unknown = np.array([ 0,0,0])
print(class_unknown)


# In[ ]:


print(individual_patched_mask.shape)
print(mask_dataset.shape)


# In[ ]:


#######generating the labels 

label = individual_patched_mask

def rgb_to_label(label):
  label_segment = np.zeros(label.shape, dtype=np.uint8)
  label_segment[np.all(label == class_urban_land , axis=-1)] = 0
  label_segment[np.all(label == class_agriculture_land, axis=-1)] = 1
  label_segment[np.all(label == class_rangeland , axis=-1)] = 2
  label_segment[np.all(label == class_forest_land, axis=-1)] = 3
  label_segment[np.all(label == class_water, axis=-1)] = 4
  label_segment[np.all(label == class_barren_land, axis=-1)] = 5
  label_segment[np.all(label == class_unknown, axis=-1)] = 6
  print(label_segment)
  label_segment = label_segment[:,:,0]
  #print(label_segment)
  return label_segment


# In[ ]:


labels = []
for i in range(mask_dataset.shape[0]):
  label = rgb_to_label(mask_dataset[i])
  labels.append(label)
  
labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)


# In[ ]:


np.unique(labels)


# In[ ]:


print("Total unique labels based on masks: ",format(np.unique(labels)))


# In[ ]:


####### using the labels to print the mask

random_image_id = random.randint(0, len(image_dataset))

plt.figure(figsize=(14,8))
plt.subplot(121)
plt.imshow(image_dataset[random_image_id])
plt.subplot(122)
# plt.imshow(mask_dataset[random_image_id])
plt.imshow(labels[random_image_id][:,:,0])


# In[ ]:


#image dataset for the model
master_training_dataset = image_dataset
print(len(master_training_dataset))
print(master_training_dataset.shape)


# In[ ]:


total_classes = len(np.unique(labels))
total_classes


# In[ ]:


get_ipython().system('pip install tensorflow')


# In[ ]:


#creating labels catagorical dataset

from tensorflow.keras.utils import to_categorical 
labels_categorical_dataset = to_categorical(labels, num_classes=total_classes)



# In[ ]:


#spliting the dataset to ttrain test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(master_training_dataset, labels_categorical_dataset, test_size =0.15, random_state = 100 )


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


image_height = X_train.shape[1]
image_width = X_train.shape[2]
image_channels = X_train.shape[3]
total_classes = y_train.shape[3]


# In[ ]:


print(image_height)
print(image_width)
print(image_channels)
print(total_classes)


# ## **Creating Models**:

# In[ ]:


get_ipython().system('pip install -U -q segmentation-models')
get_ipython().system('pip install -q tensorflow')
get_ipython().system('pip install -q keras')
get_ipython().system('pip install -q tensorflow-estimator')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm


# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate, BatchNormalization, Dropout, Lambda
from tensorflow.keras import backend as K


# In[ ]:


# creating the loss coefficient 
def jaccard_coef(y_true, y_pred):
  y_true_flatten = K.flatten(y_true)
  y_pred_flatten = K.flatten(y_pred)
  intersection = K.sum(y_true_flatten * y_pred_flatten)
  final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
  return final_coef_value


# In[ ]:


#Defining the model
def multi_unet_model(n_classes=6, image_height=256, image_width=256, image_channels=1):

  inputs = Input((image_height, image_width, image_channels))
  source_input = inputs
  #we could change 16 as Hyper parmaeter tuning and also the dropout rate which is 0.2

  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(source_input)
  c1 = Dropout(0.2)(c1)
  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(source_input)
  p1 = MaxPooling2D((2,2))(c1)

  #we could change 32 as Hyper parmaeter tuning and also the dropout rate which is 0.2

  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
  c2 = Dropout(0.2)(c2)
  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
  p2 = MaxPooling2D((2,2))(c2)

  #we could change 64 as Hyper parmaeter tuning and also the dropout rate which is 0.2

  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
  c3 = Dropout(0.2)(c3)
  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
  p3 = MaxPooling2D((2,2))(c3)

  #we could change 128 as Hyper parmaeter tuning and also the dropout rate which is 0.2

  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
  c4 = Dropout(0.2)(c4)
  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
  p4 = MaxPooling2D((2,2))(c4)

  #we could change 256 as Hyper parmaeter tuning and also the dropout rate which is 0.2
  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
  c5 = Dropout(0.2)(c5)
  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)


  u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

  u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

  u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
  c8 = Dropout(0.2)(c8)
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

  u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
  c9 = Dropout(0.2)(c9)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

  outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)

  model = Model(inputs=[inputs], outputs=[outputs])
  return model


# In[ ]:


metrics = ["accuracy", jaccard_coef]


# In[ ]:


#defined model 
def get_deep_learning_model():
  return multi_unet_model(n_classes=total_classes, image_height=image_height, image_width=image_width, image_channels=image_channels)


# In[ ]:


#call the model
model = get_deep_learning_model()
#model.get_config()


# In[ ]:


weights = [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428,0.1428]


# In[ ]:


#losses
import segmentation_models as sm
dice_loss = sm.losses.DiceLoss(class_weights = weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


# # ***Compiling the model***

# In[ ]:


import tensorflow as tf
tf.keras.backend.clear_session()
from keras import backend as K
optimizer = keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)




# In[ ]:


model.summary()


# In[ ]:


model_history = model.fit(X_train, y_train,batch_size=16,verbose=1,epochs=1000,validation_data=(X_test, y_test),shuffle=False)


# In[ ]:


history_a = model_history


# In[ ]:


print(model_history.history.keys())


# In[ ]:


# #save history
# import numpy as np
# np.save('my_history.npy',model_history.history)


# In[ ]:


# #load the saved history
# import numpy as np
# history_a=np.load('my_history.npy',allow_pickle='TRUE').item()


# In[ ]:


history_a


# In[ ]:


#plotting loss and IOU

from matplotlib import pyplot as plt
fig, axs = plt.subplots(1, 2, figsize = (18, 5))

loss = history_a.history['loss']             #loss = history_a.history['loss']
val_loss = history_a.history['val_loss']     #history_a.history['val_loss']
epochs = range(1, len(loss) + 1)

axs[0].plot(epochs, loss, 'y', label="Training Loss")
axs[0].plot(epochs, val_loss, 'r', label="Validation Loss")
axs[0].set_title("My_Unet_Training Vs Validation Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend()

jaccard_coef = history_a.history['jaccard_coef']               #history_a.history['jaccard_coef']
val_jaccard_coef = history_a.history['val_jaccard_coef']       #history_a.history['val_jaccard_coef']
epochs = range(1, len(jaccard_coef) + 1)

axs[1].plot(epochs, jaccard_coef, 'y', label="Training IoU")
axs[1].plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
axs[1].set_title("My_Unet_Training Vs Validation IoU")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss")
plt.legend()



# In[ ]:


#plotting accuracy and jc

from matplotlib import pyplot as plt
fig, axs = plt.subplots(1, 2, figsize = (18, 5))

accuracy = history_a.history['accuracy']             #loss = history_a.history['loss']
val_accuracy = history_a.history['val_accuracy']     #history_a.history['val_loss']
epochs = range(1, len(loss) + 1)

axs[0].plot(epochs, accuracy, 'y', label="Training accuracy")
axs[0].plot(epochs, val_accuracy, 'r', label="Validation accuracy")
axs[0].set_title("My_Unet_Training Vs Validation accuracy")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("accuracy")
axs[0].legend()



# In[ ]:


model_history.params


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


len(y_pred)


# In[ ]:


# y_pred
     


# In[ ]:


y_pred_argmax = np.argmax(y_pred, axis=3)


# In[ ]:


len(y_pred_argmax)


# In[ ]:


# y_pred_argmax


# In[ ]:


y_test_argmax = np.argmax(y_test, axis=3)


# In[ ]:


# y_test_argmax


# Comparing prediction results

# In[ ]:


import random


# In[ ]:


test_image_number = random.randint(0, len(X_test))

test_image = X_test[test_image_number]
ground_truth_image = y_test_argmax[test_image_number]

test_image_input = np.expand_dims(test_image, 0)

prediction = model.predict(test_image_input)
predicted_image = np.argmax(prediction, axis=3)
predicted_image = predicted_image[0,:,:]


# In[ ]:


plt.figure(figsize=(14,8))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(test_image)
plt.subplot(232)
plt.title("Original Masked image")
plt.imshow(ground_truth_image)
plt.subplot(233)
plt.title("Predicted Image")
plt.imshow(predicted_image)

     


# In[ ]:


model.save("LC_ds_segment.h5")


# In[ ]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
import os


# ### USing other models - Unet with Resnet as backbone 

# In[ ]:


# preprocess input

x_train = preprocess_input(X_train)
x_val = preprocess_input(X_test)


# In[ ]:


#LIST OF MODEL BACKBONES
# model_BACKBONE = ["resnet34" ,'inceptionv3','vgg19' ]


# In[ ]:


## using other models

# e = 1
# opt = 'Adam'
# batch_q = [16]
# histories = []

# from matplotlib import pyplot as plt

# for i in range(len(batch_q)):
#     ba_size = batch_q[i]
#     print('ba_size'+ str(ba_size))
#     for j in range(len(model_BACKBONE)):
#         #preprocessing inputs
#         preprocess_input = sm.get_preprocessing(model_BACKBONE [j])
#         x_train = preprocess_input(X_train)
#         x_val = preprocess_input(X_test)

#         M_name = "model_Unet_" + str(model_BACKBONE[j])
#         model= sm.Unet(model_BACKBONE[j], encoder_weights='imagenet', classes=total_classes, activation='softmax')
#         print(M_name)   
#         model.compile(optimizer=opt, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)
#     #     print(model.summary())
#         print(M_name)
# #         H_Name = "model_Unet_" + str(model_BACKBONE[j]) + "_history"
#         print(H_Name)
#          mymodel = model.fit(x=X_train,y=y_train,batch_size=ba_size,epochs=e,verbose=1,validation_data=(X_test, y_test),shuffle=False)
# #         histories.append[H_name]


#         #######################################################################

#         fig, axs = plt.subplots(1, 2, figsize = (18, 5))

#         loss = loss =  model.history['loss']             #history_a['loss'] 
#         val_loss =  model.history['val_loss']     #history_a['val_loss']  
#         epochs = range(1, len(loss) + 1)

#         axs[0].plot(epochs, loss, 'y', label="Training Loss")
#         axs[0].plot(epochs, val_loss, 'r', label="Validation Loss")
#         axs[0].set_title(f"My_Unet_Training Vs Validation Loss for {M_name}")
#         axs[0].set_xlabel("Epochs")
#         axs[0].set_ylabel("Loss")
#         axs[0].legend()

#         jaccard_coef =  model.history['jaccard_coef']               #history_a.history['jaccard_coef']
#         val_jaccard_coef =  model.history['val_jaccard_coef']       #history_a.history['val_jaccard_coef']
#         epochs = range(1, len(jaccard_coef) + 1)

#         axs[1].plot(epochs, jaccard_coef, 'y', label="Training IoU")
#         axs[1].plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
#         axs[1].set_title(f"My_Unet_Training Vs Validation IoU for {M_name}")
#         axs[1].set_xlabel("Epochs")
#         axs[1].set_ylabel("Loss")
#         plt.legend()

#         ####################################################################### 






#         M_name = "model_Linknet_" + str(model_BACKBONE[j])
#         model= sm.Linknet(model_BACKBONE[j], encoder_weights='imagenet', classes=total_classes, activation='softmax')
#         print(M_name)   
#         model.compile(optimizer=opt, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)
#     #     print(model.summary())
#         print(M_name)
#         H_Name = "model_Linknet_" + str(model_BACKBONE[j]) + "_history"
#         print(H_Name)
#         H_Name = model.fit(x=X_train,y=y_train,batch_size=ba_size,epochs=e,verbose=1,validation_data=(X_test, y_test),shuffle=False)
#         histories.append[H_name]


#         #######################################################################

#         fig, axs = plt.subplots(1, 2, figsize = (18, 5))

#         loss = loss = H_Name.history['loss']             #history_a['loss'] 
#         val_loss = H_Name.history['val_loss']     #history_a['val_loss']  
#         epochs = range(1, len(loss) + 1)

#         axs[0].plot(epochs, loss, 'y', label="Training Loss")
#         axs[0].plot(epochs, val_loss, 'r', label="Validation Loss")
#         axs[0].set_title(f"My_Unet_Training Vs Validation Loss for {M_name}")
#         axs[0].set_xlabel("Epochs")
#         axs[0].set_ylabel("Loss")
#         axs[0].legend()

#         jaccard_coef = H_Name.history['jaccard_coef']               #history_a.history['jaccard_coef']
#         val_jaccard_coef = H_Name.history['val_jaccard_coef']       #history_a.history['val_jaccard_coef']
#         epochs = range(1, len(jaccard_coef) + 1)

#         axs[1].plot(epochs, jaccard_coef, 'y', label="Training IoU")
#         axs[1].plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
#         axs[1].set_title(f"My_Unet_Training Vs Validation IoU for {M_name}")
#         axs[1].set_xlabel("Epochs")
#         axs[1].set_ylabel("Loss")
#         plt.legend()

#         ####################################################################### 


#         M_name = "model_Linknet_" + str(model_BACKBONE[j])
#         model= sm.FPN(model_BACKBONE[j], encoder_weights='imagenet', classes=total_classes, activation='softmax')
#         print(M_name)   
#         model.compile(optimizer=opt, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)
#     #     print(model.summary())
#         print(M_name)
#         H_Name = "model_FPN_" + str(model_BACKBONE[j]) + "_history"
#         print(H_Name)
#         H_Name = model.fit(x=X_train,y=y_train,batch_size=ba_size,epochs=e,verbose=1,validation_data=(X_test, y_test),shuffle=False)
#         histories.append[H_name]


#         #######################################################################

#         fig, axs = plt.subplots(1, 2, figsize = (18, 5))

#         loss = loss = H_Name.history['loss']             #history_a['loss'] 
#         val_loss = H_Name.history['val_loss']     #history_a['val_loss']  
#         epochs = range(1, len(loss) + 1)

#         axs[0].plot(epochs, loss, 'y', label="Training Loss")
#         axs[0].plot(epochs, val_loss, 'r', label="Validation Loss")
#         axs[0].set_title(f"My_Unet_Training Vs Validation Loss for {M_name}")
#         axs[0].set_xlabel("Epochs")
#         axs[0].set_ylabel("Loss")
#         axs[0].legend()

#         jaccard_coef = H_Name.history['jaccard_coef']               #history_a.history['jaccard_coef']
#         val_jaccard_coef = H_Name.history['val_jaccard_coef']       #history_a.history['val_jaccard_coef']
#         epochs = range(1, len(jaccard_coef) + 1)

#         axs[1].plot(epochs, jaccard_coef, 'y', label="Training IoU")
#         axs[1].plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
#         axs[1].set_title(f"My_Unet_Training Vs Validation IoU for {M_name}")
#         axs[1].set_xlabel("Epochs")
#         axs[1].set_ylabel("Loss")
#         plt.legend()

#         ####################################################################### 



# In[ ]:


# #Using Resnet as backbone

# # BACKBONE = model_BACKBONE[0]
# preprocess_input = sm.get_preprocessing(BACKBONE)


# In[ ]:


# model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=total_classes)
# model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

# print(model.summary())


# In[ ]:


# for layer in model.layers:
#   layer.trainable = False


# In[ ]:


# history_resnet34_Unet = model.fit(x=X_train,y=y_train,batch_size=16,epochs=5,verbose=1,validation_data=(X_test, y_test),shuffle=False)


# ## Unet with inceptionv3 as backbone 

# In[ ]:


BACKBONE = 'inceptionv3'
preprocess_input = sm.get_preprocessing(BACKBONE)
# preprocess input
# x_train = preprocess_input(X_train)
# x_val = preprocess_input(X_test)
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=total_classes)
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

print(model.summary())


# In[ ]:


history_inceptionv3_Unet  = model.fit(x=X_train,y=y_train,batch_size=16,epochs=5,verbose=1,validation_data=(X_test, y_test),shuffle=False)


# ## Unet with vgg19 as backbone 

# In[ ]:





# In[ ]:


BACKBONE = 'vgg19'
preprocess_input = sm.get_preprocessing(BACKBONE)
# preprocess input
# x_train = preprocess_input(X_train)
# x_val = preprocess_input(X_test)
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=total_classes)
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

print(model.summary())


# In[ ]:


history_vgg19_Unet  = model.fit(x=X_train,y=y_train,batch_size=16 ,epochs=1000,verbose=1,validation_data=(X_test, y_test),shuffle=False)


# ## Unet with mobilenetv2 as backbone 

# In[ ]:


BACKBONE = 'mobilenetv2'
preprocess_input = sm.get_preprocessing(BACKBONE)
# preprocess input
# x_train = preprocess_input(X_train)
# x_val = preprocess_input(X_test)
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=total_classes)
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

print(model.summary())


# In[ ]:


history_mobilenetv2_Unet  = model.fit(x=X_train,y=y_train,batch_size=16,epochs=1000,verbose=1,validation_data=(X_test, y_test),shuffle=False)


# In[ ]:


# model = sm.Linknet(BACKBONE, encoder_weights='imagenet', classes=total_classes)
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

print(model.summary())


# In[ ]:




