import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Concatenate, Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import regularizers
from ipdb import set_trace


dataset_path = './'
# with open(dataset_path + '/000/train.txt') as f:
    # test_filenames, test_labels = tuple(zip(*[line.split() for line in f.readlines()[1:]]))
# with open(dataset_path + '/000/test.txt') as f:
    # train_filenames, train_labels = tuple(zip(*[line.split() for line in f.readlines()[1:]]))

mySeed = 8888

# Problem 3
batch_size = 50
test_datagen_224 = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            )
train_datagen_224 = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=0
            )
test_df = pd.read_csv('dataframe/spec_bbox_train.txt', sep=',', header=None, names=['filename', 'class', 'x', 'y', 'w', 'h'])
train_df = pd.read_csv('dataframe/spec_bbox_test.txt', sep=',', header=None, names=['filename', 'class', 'x', 'y', 'w', 'h'])

# def multiflow_from_dataframe(dataframe, generator):
    # gen_x = generator.flow(dataframe)


mobileNetV2 = MobileNetV2(include_top=False,
                weights='imagenet',
                pooling=None)
for layer in mobileNetV2.layers:
    layer.trainable = False

inp = Input(shape=(224,224,3), name="input")
x = mobileNetV2(inp)
x = AveragePooling2D((7,7))(x)
x2 = x
# x = Dropout(0.45, noise_shape=None, seed=mySeed)(x)
# x = BatchNormalization(axis=-1)(x)
x = Conv2D(12, (1,1), activation='softmax', 
           kernel_regularizer=regularizers.l1(0.012),
           activity_regularizer=regularizers.l1(0.01))(x)
species = Flatten(name='species')(x)

# x2 = Dropout(0.45, noise_shape=None, seed=mySeed)(x2)
# x2 = BatchNormalization(axis=-1)(x2)
x2 = Conv2D(4, (1,1), activation='softmax', 
            kernel_regularizer=regularizers.l1(0.012),
            activity_regularizer=regularizers.l1(0.01))(x2)
bbox = Flatten(name='bbox')(x2)
output = Concatenate(axis=-1)([species, bbox])

optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Model for Species
model_species = Model(inputs=inp, outputs=species)
model_species.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
test_species_gen_224 = test_datagen_224.flow_from_dataframe(test_df, 'images/test', seed=mySeed,
                                x_col='filename', y_col='class',
                                target_size=(224, 224), class_mode='categorical', batch_size=batch_size)
train_species_gen_224 = train_datagen_224.flow_from_dataframe(train_df, 'images/train', seed=mySeed,
                                 x_col='filename', y_col='class',
                                 target_size=(224, 224), class_mode='categorical', batch_size=batch_size)
# model_species.fit_generator(generator=train_species_gen_224,
                        # validation_data=test_species_gen_224,
                        # verbose=1,
                        # epochs=80)


# Model for BBox
model_bbox = Model(inputs=inp, outputs=bbox)
model_bbox.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
test_bbox_gen_224 = test_datagen_224.flow_from_dataframe(test_df, 'images/test', seed=mySeed,
                                x_col='filename', y_col=['x','y','w','h'],
                                target_size=(224, 224), class_mode='other', batch_size=batch_size)
train_bbox_gen_224 = train_datagen_224.flow_from_dataframe(train_df, 'images/train', seed=mySeed,
                                 x_col='filename', y_col=['x','y','w','h'],
                                 target_size=(224, 224), class_mode='other', batch_size=batch_size)
# model_bbox.fit_generator(generator=train_bbox_gen_224,
                        # validation_data=test_bbox_gen_224,
                        # verbose=1,
                        # epochs=80)

# Model for Multitask
model_multi = Model(inputs=inp, outputs=output)
def combined_loss(y_true, y_pred):
    alpha = 0.5
    loss_classification = categorical_crossentropy(to_categorical(y_true[:, 0:1], 12), y_pred[:, 0:13])
    loss_regression = mean_squared_error(y_true[:, 1:], y_pred[:, 13:])
    return alpha * loss_classification + (1-alpha) * loss_regression
model_multi.compile(optimizer=optimizer, loss=combined_loss, metrics=['accuracy'])

# def combined_generator(gen_a, gen_b, output_a, output_b):
    # while True:
        # img_a, label_a = gen_a.next()
        # img_b, label_b = gen_b.next()
        # set_trace()
        # # assert np.all(img_a == img_b)
        # label = {}
        # label[output_a] = label_a
        # label[output_b] = label_b
        # yield(img_a, label)

test_multi_gen_224 = test_datagen_224.flow_from_dataframe(test_df, 'images/test', seed=mySeed,
                                x_col='filename', y_col=['class', 'x','y','w','h'],
                                target_size=(224, 224), class_mode='other', batch_size=batch_size)
train_multi_gen_224 = train_datagen_224.flow_from_dataframe(train_df, 'images/train', seed=mySeed,
                                 x_col='filename', y_col=['class', 'x','y','w','h'],
                                 target_size=(224, 224), class_mode='other', batch_size=batch_size)
model_multi.fit_generator(generator=train_multi_gen_224,
                        validation_data=test_multi_gen_224,
                        verbose=1,
                        epochs=80)

# combined_train_gen = combined_generator(train_species_gen_224, train_bbox_gen_224, 'species', 'bbox')
# combined_test_gen = combined_generator(test_species_gen_224, test_bbox_gen_224, 'species', 'bbox')
# model_multi.fit_generator(generator=combined_train_gen,
                        # validation_data=combined_test_gen,
                        # steps_per_epoch=359//batch_size,
                        # validation_steps=357/batch_size,
                        # verbose=1,
                        # epochs=80)

# optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
# optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# model.fit_generator(generator=train_gen_224,
                        # validation_data=test_gen_224,
                        # verbose=1,
                        # epochs=80)
score = model_multi.evaluate_generator(test_gen_224, verbose=1)
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
