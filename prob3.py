import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import regularizers


dataset_path = './'
test_df = pd.read_csv('species_train.txt', sep=' ', header=None, names=['filename', 'class'])
train_df = pd.read_csv('species_test.txt', sep=' ', header=None, names=['filename', 'class'])
# with open(dataset_path + '/000/train.txt') as f:
    # test_filenames, test_labels = tuple(zip(*[line.split() for line in f.readlines()[1:]]))
# with open(dataset_path + '/000/test.txt') as f:
    # train_filenames, train_labels = tuple(zip(*[line.split() for line in f.readlines()[1:]]))

from numpy.random import seed

mySeed = 8888
seed(mySeed)

# Problem 3
batch_size = 50
test_datagen_224 = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            )
test_gen_224 = test_datagen_224.flow_from_dataframe(test_df, 'images/test', seed=mySeed,
                                target_size=(224, 224), class_mode='categorical', batch_size=batch_size)

train_datagen_224 = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=0
            )
train_gen_224 = train_datagen_224.flow_from_dataframe(train_df, 'images/train', seed=mySeed,
                                 target_size=(224, 224), class_mode='categorical', batch_size=batch_size)

mobileNetV2 = MobileNetV2(include_top=False,
                weights='imagenet',
                pooling=None,
                input_shape=(224, 224, 3))

model = Sequential([
    mobileNetV2,
    AveragePooling2D((7,7)),
    Dropout(0.45, noise_shape=None, seed=mySeed), 
    Conv2D(12, (1,1), activation='softmax'),
    # BatchNormalization(axis=-1),
    # Conv2D(12, (1,1), activation='softmax'),
    # BatchNormalization(axis=-1),
    # Conv2D(12, (1,1), activation='softmax', kernel_regularizer=regularizers.l1(0.012), activity_regularizer=regularizers.l1(0.01)),
    Flatten(),
    ])

for layer in mobileNetV2.layers:
    layer.trainable = False

optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
# optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(generator=train_gen_224,
                        validation_data=test_gen_224,
                        verbose=1,
                        epochs=80)
score = model.evaluate_generator(test_gen_224, verbose=1)
print('stage 1:', score)
# for layer in mobileNetV2.layers:
    # layer.trainable = True
# model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit_generator(generator=train_gen_224,
                        # validation_data=test_gen_224,
                        # verbose=1,
                        # epochs=10)
model.save('model.h5')
model.save_weights('weights.h5')

# score = model.evaluate_generator(test_gen_224, verbose=1)
# print(score)
