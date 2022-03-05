---
layout: post
title: Classification is the Name of the Game!
---

## Image Classification!

In this blogpost we're going to be playing around with image clasification in tensorflow. We'll be engaging with the most important of questions: cats vs dogs (dogs of course being the right answer). We'll be building a number of models of increasing complexity and looking at how each of those models score in terms of classifying a dataset of cat and dog images. 

To get started lets import the necessary packages and load in the data from the given URL. Then, we seperate the train and validation sets from the dataset and use that to construct our validation, test, and, train datasets. The test dataset will comprise 1/5th of our validation dataset.


```python
import os
import tensorflow as tf
from tensorflow.keras import utils, datasets, layers, models
import matplotlib.pyplot as plt
```


```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
```

    Found 2000 files belonging to 2 classes.
    Found 1000 files belonging to 2 classes.


Next lets create a function to plot a row of 3 cats and a row of 3 dogs so we can start to visualize our dataset. 


```python
def plot_catsndogs(train_dataset):
    '''
    takes in a dataset of cat and dog images, 
    and then plots a row of 3 cats and then a row of 3 dogs.
    
    @args
    train_dataset: BatchDataset of car and dog images
    
    @return-none
    '''
    #create the plot
    plt.figure(figsize=(10, 10))
    
    #use .take to retrive 1 batch of images from our dataset
    for images, labels in train_dataset.take(1):
        #create sets of cat images by filtering by label and do the dame for dogs
        cats=images[labels==0] 
        dogs=images[labels==1]
        
        #plot 3 images from our list of cat images.
        for i in range(3):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(cats[i].numpy().astype("uint8"))
            plt.title('cats')
            plt.axis("off")
            
            #plot three images from our list of dog images.
        for i in range(3):
            ax = plt.subplot(2, 3, i + 4)
            plt.imshow(dogs[i].numpy().astype("uint8"))
            plt.title('dogs')
            plt.axis("off")
```

Now lets use the function we just defined to take a look at those adorable cats and dogs!


```python
plot_catsndogs(train_dataset)
```


    
![output_8_0.png]({{ site.baseurl }}/images/output_8_0.png)
    


Look how cute they are! Aren't you glad you get to look at them and learn at the same time? A real win-win.

Next we'll run the following code block in order to allow us to rapidly read data.


```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

Now we're going to create an iterator to iterate over the data and then use that to calculate the number of cat images and dog images we have in our data. Knowing this helps us to figure out what the base model would do. I.e, if a model just guessed the most common class every time, how accurate would it be?


```python
#create the iterator
labels_iterator= train_dataset.unbatch().map(lambda image, label: label).as_numpy_iterator()

#create variables for the number of cats and dogs
cat_labels=0
dog_labels=0

#run through each image and save the number of cat/dog images to our variables
for label in labels_iterator:
    if(label==1):
        dog_labels+=1
    else:
        cat_labels+=1
        
print(cat_labels)
print(dog_labels)
```

    1000
    1000


In this case we have 1000 images of cats and 1000 images of dogs so the baseline would randomly guess one or the other and would habe a baseline accuracy of 50%.

## Model 1

Now we get to build our first model! Below we define the layer sturcture using the tensorflow keras sequential API. We start with convolutional 2d layers combined with maxpooling layers to help us get "zoom out" more quickly to get higher levels of abstraction. Then we use a dropout layer to prevent overfitting, a flatten layer to change our data from 2d to 1d, and 1 dense layers to add more abstraction and then one more dense layer to output the final model class prediction.


```python
#build the model using the keras sequential API
model1=tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
])
```

Finally we compile the model using the adam optimizer and SparseCategoricalCrossentropy as our loss function. We include the accuracy metric so we can easily track the training and validation accuracy as we train the model. Then we train for 20 epochs on our training dataset, saving the results to the history variable for later inspection.



```python
#compile the model
model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
history = model1.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 39s 596ms/step - loss: 7.3819 - accuracy: 0.4825 - val_loss: 0.6931 - val_accuracy: 0.5037
    Epoch 2/20
    63/63 [==============================] - 35s 551ms/step - loss: 0.6923 - accuracy: 0.5130 - val_loss: 0.6873 - val_accuracy: 0.5656
    Epoch 3/20
    63/63 [==============================] - 35s 554ms/step - loss: 0.6936 - accuracy: 0.5085 - val_loss: 0.6929 - val_accuracy: 0.5025
    Epoch 4/20
    63/63 [==============================] - 36s 560ms/step - loss: 0.6897 - accuracy: 0.5300 - val_loss: 0.6931 - val_accuracy: 0.4988
    Epoch 5/20
    63/63 [==============================] - 35s 549ms/step - loss: 0.6886 - accuracy: 0.5250 - val_loss: 0.6760 - val_accuracy: 0.5854
    Epoch 6/20
    63/63 [==============================] - 36s 566ms/step - loss: 0.6755 - accuracy: 0.5865 - val_loss: 0.6846 - val_accuracy: 0.5594
    Epoch 7/20
    63/63 [==============================] - 34s 541ms/step - loss: 0.7142 - accuracy: 0.5885 - val_loss: 0.7283 - val_accuracy: 0.5408
    Epoch 8/20
    63/63 [==============================] - 35s 552ms/step - loss: 0.6403 - accuracy: 0.6295 - val_loss: 0.6626 - val_accuracy: 0.6163
    Epoch 9/20
    63/63 [==============================] - 35s 548ms/step - loss: 0.5593 - accuracy: 0.7030 - val_loss: 0.6737 - val_accuracy: 0.6250
    Epoch 10/20
    63/63 [==============================] - 34s 544ms/step - loss: 0.4667 - accuracy: 0.7805 - val_loss: 0.6735 - val_accuracy: 0.6832
    Epoch 11/20
    63/63 [==============================] - 34s 544ms/step - loss: 0.4359 - accuracy: 0.7990 - val_loss: 0.7217 - val_accuracy: 0.6448
    Epoch 12/20
    63/63 [==============================] - 35s 546ms/step - loss: 0.3491 - accuracy: 0.8445 - val_loss: 0.7587 - val_accuracy: 0.7054
    Epoch 13/20
    63/63 [==============================] - 35s 544ms/step - loss: 0.2643 - accuracy: 0.8905 - val_loss: 0.8479 - val_accuracy: 0.6634
    Epoch 14/20
    63/63 [==============================] - 34s 543ms/step - loss: 0.2155 - accuracy: 0.9150 - val_loss: 1.0954 - val_accuracy: 0.6782
    Epoch 15/20
    63/63 [==============================] - 35s 545ms/step - loss: 0.2036 - accuracy: 0.9210 - val_loss: 1.0384 - val_accuracy: 0.6844
    Epoch 16/20
    63/63 [==============================] - 35s 554ms/step - loss: 0.1692 - accuracy: 0.9355 - val_loss: 1.1544 - val_accuracy: 0.6869
    Epoch 17/20
    63/63 [==============================] - 35s 546ms/step - loss: 0.1435 - accuracy: 0.9460 - val_loss: 1.1144 - val_accuracy: 0.6844
    Epoch 18/20
    63/63 [==============================] - 35s 546ms/step - loss: 0.0923 - accuracy: 0.9630 - val_loss: 1.4148 - val_accuracy: 0.6955
    Epoch 19/20
    63/63 [==============================] - 35s 549ms/step - loss: 0.1147 - accuracy: 0.9550 - val_loss: 1.3670 - val_accuracy: 0.6733
    Epoch 20/20
    63/63 [==============================] - 35s 544ms/step - loss: 0.1265 - accuracy: 0.9575 - val_loss: 1.3349 - val_accuracy: 0.6894


Finally, use the history variable to plot the training and validation accuracy across training epochs.


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fece98022b0>




![output_21_1.png]({{ site.baseurl }}/images/output_21_1.png)    

    


#### 1. The validation accuraxy of my model stabilized between 67% and 68%.
2. That represents about a 18% increase over baseline. 
3. However it is clear that the model is overfitted since training accuracy reached far higher at nearly 100% accuracy.

## Model 2- With Data Augmentation

For the next model we will use a similar setup except we will first modify the training images by randomly flipping and rotating them in order to help the model learn invariant features in our images.


```python
#create a model that randomly flips the image horizontally and vertically
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal_and_vertical'),
])
```

We will visulalize the effect of the above model by running the same image through it multiple times and plotting it.


```python
#retreive an image from the dataset
for image, _ in train_dataset.take(1):
      plt.figure(figsize=(10, 10))
      first_image = image[0]
    #plot that same image 9 times, each time using the data_augmentation model
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
```


![output_27_0.png]({{ site.baseurl }}/images/output_27_0.png)     



As we can see, that model took the image and randomly flipped it, sometimes horizontally and sometimes vertically. Next we will do the exact same thing except instead of flipping the image the new model will use the RandomRotation method to occasionally rotate the read in image. We will again plot an image 9 times using the model.


```python
#create a new model using the random rotation layer
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(0.5),
])

#retrieve image from the dataset
for image, _ in train_dataset.take(1):
      plt.figure(figsize=(10, 10))
      first_image = image[0]
    #plot that same image 9 times, each time using the data_augmentation model
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
```


    
![output_29_0.png]({{ site.baseurl }}/images/output_29_0.png) 
    


Finally we build the model architecture. As you can see it is the same as the first model except for the addition of the RandomFlip and RandomRotation layers at the beggining of the model layering. We than compile and train our model as before, then plot the results.


```python
model2=tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
])
```


```python
model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
history2 = model2.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 40s 613ms/step - loss: 5.3348 - accuracy: 0.5245 - val_loss: 0.6963 - val_accuracy: 0.5087
    Epoch 2/20
    63/63 [==============================] - 38s 607ms/step - loss: 0.6902 - accuracy: 0.5510 - val_loss: 0.6922 - val_accuracy: 0.5446
    Epoch 3/20
    63/63 [==============================] - 38s 599ms/step - loss: 0.6929 - accuracy: 0.5360 - val_loss: 0.6908 - val_accuracy: 0.5495
    Epoch 4/20
    63/63 [==============================] - 38s 600ms/step - loss: 0.6849 - accuracy: 0.5660 - val_loss: 0.6868 - val_accuracy: 0.5644
    Epoch 5/20
    63/63 [==============================] - 38s 601ms/step - loss: 0.6771 - accuracy: 0.5460 - val_loss: 0.6651 - val_accuracy: 0.5693
    Epoch 6/20
    63/63 [==============================] - 39s 607ms/step - loss: 0.6668 - accuracy: 0.5665 - val_loss: 0.6775 - val_accuracy: 0.5730
    Epoch 7/20
    63/63 [==============================] - 38s 604ms/step - loss: 0.6794 - accuracy: 0.5845 - val_loss: 0.6729 - val_accuracy: 0.5520
    Epoch 8/20
    63/63 [==============================] - 39s 619ms/step - loss: 0.6584 - accuracy: 0.6115 - val_loss: 0.6588 - val_accuracy: 0.5953
    Epoch 9/20
    63/63 [==============================] - 38s 598ms/step - loss: 0.6989 - accuracy: 0.5245 - val_loss: 0.6959 - val_accuracy: 0.5136
    Epoch 10/20
    63/63 [==============================] - 38s 601ms/step - loss: 0.6856 - accuracy: 0.5585 - val_loss: 0.6952 - val_accuracy: 0.5136
    Epoch 11/20
    63/63 [==============================] - 39s 610ms/step - loss: 0.6857 - accuracy: 0.5605 - val_loss: 0.6986 - val_accuracy: 0.5334
    Epoch 12/20
    63/63 [==============================] - 38s 603ms/step - loss: 0.6761 - accuracy: 0.5955 - val_loss: 0.6764 - val_accuracy: 0.5755
    Epoch 13/20
    63/63 [==============================] - 39s 610ms/step - loss: 0.6879 - accuracy: 0.5615 - val_loss: 0.6932 - val_accuracy: 0.5347
    Epoch 14/20
    63/63 [==============================] - 39s 619ms/step - loss: 0.6871 - accuracy: 0.5510 - val_loss: 0.6909 - val_accuracy: 0.5408
    Epoch 15/20
    63/63 [==============================] - 38s 600ms/step - loss: 0.6930 - accuracy: 0.5225 - val_loss: 0.6914 - val_accuracy: 0.5272
    Epoch 16/20
    63/63 [==============================] - 38s 599ms/step - loss: 0.6779 - accuracy: 0.5625 - val_loss: 0.6703 - val_accuracy: 0.5941
    Epoch 17/20
    63/63 [==============================] - 38s 602ms/step - loss: 0.6796 - accuracy: 0.5870 - val_loss: 0.6756 - val_accuracy: 0.5879
    Epoch 18/20
    63/63 [==============================] - 38s 604ms/step - loss: 0.6650 - accuracy: 0.5995 - val_loss: 0.6447 - val_accuracy: 0.6101
    Epoch 19/20
    63/63 [==============================] - 38s 597ms/step - loss: 0.6501 - accuracy: 0.6235 - val_loss: 0.6438 - val_accuracy: 0.6238
    Epoch 20/20
    63/63 [==============================] - 38s 601ms/step - loss: 0.6597 - accuracy: 0.6010 - val_loss: 0.6389 - val_accuracy: 0.6399



```python
plt.plot(history2.history["accuracy"], label = "training")
plt.plot(history2.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fec81832d60>




    
![output_34_1.png]({{ site.baseurl }}/images/output_34_1.png) 
    


#### 1. I was able to acheive model validation accuracy of ~ 63%. However it appears that better validation accuracy might be acheived with continued training.
2. In comparison to model 1, this model supprisingly had lower validation accuracy than model 1. The training and validation scores were also much more variable between epochs. 
3. I do no observe nearly as much overfitting in this model as the test and validation scores stayed close across almost all epochs

## Model 3- With Data Preprocessing

This model is again very similar to the 2 above with the addition of a preprocessing layer. This layer takes out RGB values from 0 to 255 and scales them to be between -1 and 1 which can help the training process run faster. All the other steps are identicle to the above models.


```python
#create the preprocessor layer
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])
```


```python
#build the model architecture, adding the preprocessor at the beggining
model3=tf.keras.Sequential([
    preprocessor,
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
])
```


```python
#compile the model
model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
#train the model 
history3 = model3.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 42s 650ms/step - loss: 0.7048 - accuracy: 0.5030 - val_loss: 0.6929 - val_accuracy: 0.4913
    Epoch 2/20
    63/63 [==============================] - 43s 681ms/step - loss: 0.6908 - accuracy: 0.5005 - val_loss: 0.6802 - val_accuracy: 0.5705
    Epoch 3/20
    63/63 [==============================] - 42s 661ms/step - loss: 0.6761 - accuracy: 0.5585 - val_loss: 0.6451 - val_accuracy: 0.5978
    Epoch 4/20
    63/63 [==============================] - 41s 640ms/step - loss: 0.6495 - accuracy: 0.6040 - val_loss: 0.6545 - val_accuracy: 0.6386
    Epoch 5/20
    63/63 [==============================] - 43s 680ms/step - loss: 0.6383 - accuracy: 0.6320 - val_loss: 0.6297 - val_accuracy: 0.6646
    Epoch 6/20
    63/63 [==============================] - 41s 645ms/step - loss: 0.6202 - accuracy: 0.6475 - val_loss: 0.5998 - val_accuracy: 0.6770
    Epoch 7/20
    63/63 [==============================] - 44s 691ms/step - loss: 0.6124 - accuracy: 0.6645 - val_loss: 0.6171 - val_accuracy: 0.6572
    Epoch 8/20
    63/63 [==============================] - 42s 653ms/step - loss: 0.5931 - accuracy: 0.6840 - val_loss: 0.5977 - val_accuracy: 0.6609
    Epoch 9/20
    63/63 [==============================] - 39s 613ms/step - loss: 0.5775 - accuracy: 0.6805 - val_loss: 0.5956 - val_accuracy: 0.6708
    Epoch 10/20
    63/63 [==============================] - 43s 681ms/step - loss: 0.5744 - accuracy: 0.6965 - val_loss: 0.5867 - val_accuracy: 0.6918
    Epoch 11/20
    63/63 [==============================] - 39s 614ms/step - loss: 0.5640 - accuracy: 0.7045 - val_loss: 0.5444 - val_accuracy: 0.7116
    Epoch 12/20
    63/63 [==============================] - 43s 685ms/step - loss: 0.5552 - accuracy: 0.7175 - val_loss: 0.5822 - val_accuracy: 0.7017
    Epoch 13/20
    63/63 [==============================] - 44s 686ms/step - loss: 0.5509 - accuracy: 0.7185 - val_loss: 0.5507 - val_accuracy: 0.6968
    Epoch 14/20
    63/63 [==============================] - 47s 742ms/step - loss: 0.5538 - accuracy: 0.7075 - val_loss: 0.5584 - val_accuracy: 0.7141
    Epoch 15/20
    63/63 [==============================] - 46s 727ms/step - loss: 0.5359 - accuracy: 0.7260 - val_loss: 0.5805 - val_accuracy: 0.6980
    Epoch 16/20
    63/63 [==============================] - 41s 653ms/step - loss: 0.5287 - accuracy: 0.7340 - val_loss: 0.5942 - val_accuracy: 0.6745
    Epoch 17/20
    63/63 [==============================] - 45s 713ms/step - loss: 0.5475 - accuracy: 0.7150 - val_loss: 0.5578 - val_accuracy: 0.7092
    Epoch 18/20
    63/63 [==============================] - 40s 630ms/step - loss: 0.5205 - accuracy: 0.7340 - val_loss: 0.5350 - val_accuracy: 0.7228
    Epoch 19/20
    63/63 [==============================] - 38s 606ms/step - loss: 0.5231 - accuracy: 0.7365 - val_loss: 0.5853 - val_accuracy: 0.6782
    Epoch 20/20
    63/63 [==============================] - 39s 616ms/step - loss: 0.5194 - accuracy: 0.7410 - val_loss: 0.5422 - val_accuracy: 0.7203



```python
#plot the training history
plt.plot(history3.history["accuracy"], label = "training")
plt.plot(history3.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fec882f6fa0>




    
![output_42_1.png]({{ site.baseurl }}/images/output_42_1.png) 
    


#### 1. I was able to acheive model validation accuracy of ~ 72%. However it appears that better validation accuracy might be acheived with continued training as the accuracy is still going up.
2. In comparison to model 1 and 2, this model was more accurate and has significatly less variation between epochs than model 2.
3. I do no observe significant overfitting in this model however, more overfitting is noted toward the end of the training process.

## Model 4- With transfer learning

Our final model incorrporates transfer learning. This means that we download a pretrained model and then build opon it to construct our model. This means that we can take advantage of models that have already been trained on much larger datasets than would be possible on our personal computer. 

First, lets download the MobileNetV2 pretrained image classification model from keras and use it to contruct a model layer.


```python
#download the model, setting include_top to false so we
#clip off the origional classification layer, we'll add our own taylored for 
#this specific task.
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
#freeze the base layers
base_model.trainable = False

#use the model to create a layer we can incorporate into our architecture
i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])
```

Now we'll use that pretrained layer in place of our 2d Convolutional layers from before. On top of that all we have to add is a flatten layer and 1 Dense layer for our output and were all set to compile and train!


```python
model4=tf.keras.Sequential([
    preprocessor,
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    base_model_layer,
    layers.Flatten(),
    layers.Dense(2)
])
```


```python
model4.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
model4.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model_2 (Functional)        (None, 160, 160, 3)       0         
                                                                     
     random_rotation_1 (RandomRo  (None, 160, 160, 3)      0         
     tation)                                                         
                                                                     
     random_flip_1 (RandomFlip)  (None, 160, 160, 3)       0         
                                                                     
     model_3 (Functional)        (None, 5, 5, 1280)        2257984   
                                                                     
     flatten_1 (Flatten)         (None, 32000)             0         
                                                                     
     dense_2 (Dense)             (None, 2)                 64002     
                                                                     
    =================================================================
    Total params: 2,321,986
    Trainable params: 64,002
    Non-trainable params: 2,257,984
    _________________________________________________________________


As we can see from the 2257984 parameters in the base_model, theres a lot of hidden complexity in the model that we downloaded. Now we're ready to train.



```python
history4 = model4.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 39s 584ms/step - loss: 1.0082 - accuracy: 0.8455 - val_loss: 0.3151 - val_accuracy: 0.9443
    Epoch 2/20
    63/63 [==============================] - 36s 570ms/step - loss: 0.4978 - accuracy: 0.9205 - val_loss: 0.2141 - val_accuracy: 0.9666
    Epoch 3/20
    63/63 [==============================] - 36s 577ms/step - loss: 0.6645 - accuracy: 0.9100 - val_loss: 0.2366 - val_accuracy: 0.9691
    Epoch 4/20
    63/63 [==============================] - 36s 570ms/step - loss: 0.6540 - accuracy: 0.9205 - val_loss: 0.3064 - val_accuracy: 0.9629
    Epoch 5/20
    63/63 [==============================] - 36s 565ms/step - loss: 0.5515 - accuracy: 0.9240 - val_loss: 0.2401 - val_accuracy: 0.9678
    Epoch 6/20
    63/63 [==============================] - 36s 575ms/step - loss: 0.4740 - accuracy: 0.9365 - val_loss: 0.2025 - val_accuracy: 0.9715
    Epoch 7/20
    63/63 [==============================] - 36s 575ms/step - loss: 0.5952 - accuracy: 0.9300 - val_loss: 0.3690 - val_accuracy: 0.9604
    Epoch 8/20
    63/63 [==============================] - 36s 570ms/step - loss: 0.6571 - accuracy: 0.9365 - val_loss: 0.2890 - val_accuracy: 0.9703
    Epoch 9/20
    63/63 [==============================] - 36s 568ms/step - loss: 0.6293 - accuracy: 0.9330 - val_loss: 0.3068 - val_accuracy: 0.9678
    Epoch 10/20
    63/63 [==============================] - 36s 577ms/step - loss: 0.6926 - accuracy: 0.9305 - val_loss: 0.3561 - val_accuracy: 0.9592
    Epoch 11/20
    63/63 [==============================] - 36s 570ms/step - loss: 0.5698 - accuracy: 0.9450 - val_loss: 0.3175 - val_accuracy: 0.9616
    Epoch 12/20
    63/63 [==============================] - 36s 568ms/step - loss: 0.5273 - accuracy: 0.9425 - val_loss: 0.4844 - val_accuracy: 0.9554
    Epoch 13/20
    63/63 [==============================] - 37s 589ms/step - loss: 0.7646 - accuracy: 0.9335 - val_loss: 0.3948 - val_accuracy: 0.9703
    Epoch 14/20
    63/63 [==============================] - 36s 569ms/step - loss: 0.5606 - accuracy: 0.9470 - val_loss: 0.8823 - val_accuracy: 0.9381
    Epoch 15/20
    63/63 [==============================] - 36s 567ms/step - loss: 0.7099 - accuracy: 0.9390 - val_loss: 0.5830 - val_accuracy: 0.9678
    Epoch 16/20
    63/63 [==============================] - 36s 573ms/step - loss: 0.4951 - accuracy: 0.9480 - val_loss: 0.6462 - val_accuracy: 0.9554
    Epoch 17/20
    63/63 [==============================] - 37s 585ms/step - loss: 0.5372 - accuracy: 0.9485 - val_loss: 0.4396 - val_accuracy: 0.9604
    Epoch 18/20
    63/63 [==============================] - 36s 574ms/step - loss: 0.5422 - accuracy: 0.9525 - val_loss: 0.4896 - val_accuracy: 0.9616
    Epoch 19/20
    63/63 [==============================] - 44s 705ms/step - loss: 0.6037 - accuracy: 0.9435 - val_loss: 0.5086 - val_accuracy: 0.9604
    Epoch 20/20
    63/63 [==============================] - 47s 737ms/step - loss: 0.9334 - accuracy: 0.9245 - val_loss: 0.6543 - val_accuracy: 0.9517



```python
#plot the training accuracy history
plt.plot(history4.history["accuracy"], label = "training")
plt.plot(history4.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fec886f4940>




    
![output_53_1.png]({{ site.baseurl }}/images/output_53_1.png) 
    


#### 1. This model produced validation accuracy scores of between 95-96%. 
2. This is much higher than the accuracy produced by any of the previous models.
3. This model does not appear to be overfitting to any great extent. In fact, the validation accuracy tends to be higher than the training dataset. 

## Scoring on Test Data

Now we score out most accurate model (number 4) on our test data to see how it performs.


```python
test_acc=model4.evaluate(test_dataset)
```

    6/6 [==============================] - 3s 491ms/step - loss: 0.2773 - accuracy: 0.9896



```python
print(test_acc)
print(model4.metrics_names)
```

    [0.27733954787254333, 0.9895833134651184]
    ['loss', 'accuracy']


As we can see above, this model produced a test data accuracy of ~99%! This is very high compaired to the other models we produced in this blog post and highlights the power of harnessing pretrained models through transfer learning. 

You did it! Congrats! On to bigger and better adventures in transfer learning.