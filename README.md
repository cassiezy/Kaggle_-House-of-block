# Kaggle House of block

#### This project is one of the assignments for [T81-558: Applications of Deep Neural Netw1orks](https://sites.wustl.edu/jeffheaton/t81-558/) at [Washington University in St. Louis](https://www.wustl.edu). See more detail at [Kaggle In-Class House of Blocks Competition](https://www.kaggle.com/c/applications-of-deep-learning-wustl-fall-2020) 
#### The data set is made up of many different buildings, built from a simulated set of wooden building blocks. The task is to predict the structural stability of block buildings. See example of block building below:
<img src="https://github.com/cassiezy/Kaggle_-House-of-block/blob/master/blocks-unstable.png" width = '400'>

### 1. Preprocessing image
#### A. Lightening
#### B. Sharpening
#### C. Cropping

### 2. Model Setup and Tuning

read data set
```Python
PATH = "/content/drive/MyDrive/"
PATH_TRAIN = os.path.join(PATH, "train.csv")
PATH_TEST = os.path.join(PATH, "test.csv")
PATH = "/content/lighter-png/"
```
add the image filenames to the dataset. For example, image ID #1 will correspond to 1.jpg.
```Python
df_train = pd.read_csv(PATH_TRAIN)
df_test = pd.read_csv(PATH_TEST)

df_train = df_train[df_train.id != 1300]

df_train['filename'] = df_train["id"].astype(str)+".png"
df_train['stable'] = df_train['stable'].astype(str)

df_test['filename'] = df_test["id"].astype(str)+".png"
```

This data is fairly well balanced.
```Python
df_train.stable.value_counts().plot(kind='bar')
plt.title('Labels counts')
plt.xlabel('Stable')
plt.ylabel('Count')
plt.show()
```

train_test split
```Python
TRAIN_PCT = 0.9
TRAIN_CUT = int(len(df_train) * TRAIN_PCT)

df_train_cut = df_train[0:TRAIN_CUT]
df_validate_cut = df_train[TRAIN_CUT:]

print(f"Training size: {len(df_train_cut)}")
print(f"Validate size: {len(df_validate_cut)}")
```
Training size: 36882

Validate size: 4099


#### Start tuning
Next, we create the generators that will provide the images to the neural network as it is trained.  We normalize the images so that the RGB colors between 0-255 become ratios between 0 and 1.  We also use the **flow_from_dataframe** generator to connect the Pandas dataframe to the actual image files. We see here a straightforward implementation; you might also wish to use some of the image transformations provided by the data generator.

The **HEIGHT** and **WIDTH** constants specify the dimensions that the image will be scaled (or expanded) to. It is probably not a good idea to expand the images.

We now create the neural network and fit it. Some essential concepts are going on here.

+ **Batch Size** - The number of training samples that should be evaluated per training step. Smaller batch sizes, or mini-batches, are generally preferred.
+ **Step** - A training step is one complete run over the batch. At the end of a step, the weights are updated, and the neural network learns.
+ **Epoch** - An arbitrary point at which to measure results or checkpoint the model. Generally, an epoch is one complete pass over the training set. However, when generators are used, the training set size is theoretically infinite. Because of this, we set a **steps_per_epoch** parameter.
+ **Validation steps** - The validation set may also be infinite; because of this, we must specify how many steps we wish to validate at the end of each Epoch.

```Python
HEIGHT = 331
WIDTH = 331

training_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip=True,
  fill_mode='nearest')

train_generator = training_datagen.flow_from_dataframe(
        dataframe=df_train_cut,
        color_mode = 'rgb',
        directory=PATH,
        x_col="filename",
        y_col="stable",
        target_size=(HEIGHT, WIDTH),
        batch_size=32,
        class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = validation_datagen.flow_from_dataframe(
        dataframe=df_validate_cut,
        directory=PATH,
        x_col="filename",
        y_col="stable",
        target_size=(HEIGHT, WIDTH),
        batch_size=32,
        class_mode='categorical')
```
Found 36882 validated image filenames belonging to 2 classes.

Found 4099 validated image filenames belonging to 2 classes.

#### Learning rate schedule
```Python
EPOCHS = 16

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005 * 8 #tpu_strategy.num_replicas_in_sync
rampup_epochs = 5
sustain_epochs = 0
exp_decay = .8

def lrfn(epoch):
  if epoch < rampup_epochs:
    return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
  elif epoch < rampup_epochs + sustain_epochs:
    return max_lr
  else:
    return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

rang = np.arange(EPOCHS)
y = [lrfn(x) for x in rang]
plt.plot(rang, y)
print('Learning rate per epoch:')
```

```Python
for x_batch,y_batch in train_generator:
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(x_batch[i].reshape(331,331,3), cmap='gray')
    plt.show()
```

#### Mobilenet performs best
```Python
base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(331,331,3)) 
base_model.summary()
```

```Python
x=base_model.output
x=GlobalAveragePooling2D()(x)
preds=Dense(2,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

# len_net = len(base_model.layers)

for layer in model.layers[:-2]:
  layer.trainable=False
for layer in model.layers[-2:]:
  layer.trainable=True

model.summary()
```

```Python
train_steps = train_generator.n//train_generator.batch_size
validation_steps = val_generator.n//val_generator.batch_size

model.compile(loss = 'categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=1, mode='auto',
        restore_best_weights=True)
callback_list = [monitor, lr_callback]
history = model.fit(train_generator, epochs=50, steps_per_epoch=1000,
                    validation_data = val_generator, 
                    verbose = 1, validation_steps = 20, callbacks=callback_list)
```

## Summary
+ MobileNet and Xception fit the data better
+ Training most of the layers in pretrained layers brings more improvements to the accuracy than training only the two added layers
+ Larger input image size (~331) leads to better accuracy than smaller size (~169)
+ Small batch sizes (8, 12, or 16) work much better
+ Full-step seems to be a necessity for reducing log loss to below 0.01
