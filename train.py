import os, sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.applications import MobileNetV2
from efficientnet.tfkeras import EfficientNetB1

def tf_data_generator(generator, input_shape):
    num_class = generator.num_classes
    tf_generator = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None
                        , input_shape[0]
                        , input_shape[1]
                        , input_shape[2]]
                       ,[None, num_class])
    )
    return tf_generator

def train_model(dataset_path: str, epochTotal: int, model_path: str):
    # Define Input Parameters
    dim = (160, 160)
    # dim = (456, 456)
    channel = (3, )
    input_shape = dim + channel
    #batch size
    batch_size = 16

    #Epoch
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    # binary = [1,0,0,0,0] [0,1,0,0,0] [0,0,1,0,0] [0,0,0,1,0] [0,0,0,0,1]
    # categorical = 1,2,3,4,5
    train_path = dataset_path+'/train/'
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=dim,
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)
    validation_path = dataset_path+'/validation/'
    val_generator = val_datagen.flow_from_directory(validation_path,
                                                    target_size=dim,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)
    test_path = dataset_path+'/test/'
    test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size=dim,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

    num_class = test_generator.num_classes
    labels = train_generator.class_indices.keys()

    train_data = tf_data_generator(train_generator, input_shape)
    test_data = tf_data_generator(test_generator, input_shape)
    val_data = tf_data_generator(val_generator, input_shape)

    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=input_shape))

    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # model.summary()

    # get base models
    base_model = MobileNetV2(
        input_shape= input_shape,
        include_top=False,
        weights='imagenet',
        classes=num_class,
    )

    #Adding custom layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation="relu")(x)

    predictions = layers.Dense(num_class, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.summary()

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # get base models
    base_model = EfficientNetB1(
        input_shape=input_shape,
        include_top=False,
        weights='noisy-student',
        classes=num_class,
    )

    #Adding custom layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation="relu")(x)

    predictions = layers.Dense(num_class, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.summary()
    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(x=train_data,
            steps_per_epoch=len(train_generator),
            epochs=int(epochTotal),
            validation_data=val_data,
            validation_steps=len(val_generator), 
            shuffle=True,
            verbose = 1)

    if model_path != "":
        model_path = "model"
    SAVE_MODEL_NAME = "model.h5"
    save_model_path = os.path.join(model_path, SAVE_MODEL_NAME)

    if os.path.exists(os.path.join(model_path)) == False:
        os.makedirs(os.path.join(model_path))
        
    # print('Saving Model At {}...'.format(save_model_path))
    model.save(save_model_path,include_optimizer=False)   

    loss, acc = model.evaluate(train_data, steps=len(train_generator), verbose=0)
    # print('Accuracy on training data: {:.4f} \nLoss on training data: {:.4f}'.format(acc,loss),'\n')
    lossTest, accTest = model.evaluate(test_data, steps=len(test_generator), verbose=0)
    # print('Accuracy on test data: {:.4f} \nLoss on test data: {:.4f}'.format(acc,loss),'\n') 
    
    result = {
        "label": labels,
        "train" : {
            "loss" : loss,
            "accuracy" : acc
        },
        "test" : {
            "loss" : lossTest,
            "accuracy" : accTest
        },
        "model_path": save_model_path,
        "history": {
            "accuracy": history.history['accuracy'],
            "loss": history.history['loss']
        }
    }
    return result
result = train_model("dataset", sys.argv[1], sys.argv[2])
print(result)