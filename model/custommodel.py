import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

class CustomModel:
    def __init__(self):
        self.basemodel = None
        self.predictions = None


    # base model 생성 종류는 'VGG16', 'VGG19', 'ResNet152', 'InceptionResNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'MobileNet'
    def get_basemodel(self, model_name, weights, include_top, input_shape):
        if model_name == 'VGG16':
            self.basemodel = tf.keras.applications.VGG16(weights=weights, include_top=include_top, input_shape=input_shape)
        elif model_name == 'VGG19':
            self.basemodel = tf.keras.applications.VGG19(weights=weights, include_top=include_top, input_shape=input_shape)
        elif model_name == 'ResNet152':
            self.basemodel = tf.keras.applications.ResNet152(weights=weights,include_top=include_top, input_shape=input_shape)
        elif model_name == 'InceptionResNetV2':
            self.basemodel = tf.keras.applications.InceptionResNetV2(weights=weights, include_top=include_top, input_shape=input_shape)
        elif model_name == 'DenseNet121':
            self.basemodel = tf.keras.applications.DenseNet121(weights=weights, include_top=include_top, input_shape=input_shape)
        elif model_name == 'DenseNet169':
            self.basemodel = tf.keras.applications.DenseNet169(weights=weights, include_top=include_top, input_shape=input_shape)
        elif model_name == 'DenseNet201':
            self.basemodel = tf.keras.applications.DenseNet201(weights=weights, include_top=include_top, input_shape=input_shape)
        elif model_name == 'MobileNet':
            self.basemodel = tf.keras.applications.MobileNet(weights=weights, include_top=include_top, input_shape=input_shape)
        else:
            raise ValueError("Unsupported model")

        return self.basemodel


    # base model에 FClayer, BatchNormalization 추가
    def add_layer(self, fc_unit_list, num_classes):

        for layer in self.basemodel.layers:
            layer.trainable = False

        x = Flatten()(self.basemodel.output)

        for unit in fc_unit_list:
            x = Dense(unit, activation='relu')(x)
            x = BatchNormalization()(x)

        self.predictions = Dense(num_classes, activation='softmax')(x)

        return self.predictions


    # layer 추가한 head model
    def get_headmodel(self, optimizer, loss, metrics, train_ds, val_ds, epochs):
        model = Model(inputs=self.basemodel.input, outputs=self.predictions)
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)

        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[es])

        return model, history
