from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import keras.backend as K


def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def img_preprocess(img):
    img = (img + 1.) * 127.5
    return preprocess_input(img)


def perceptual_loss(y_true, y_pred):
    y_true = img_preprocess(y_true)
    y_pred = img_preprocess(y_pred)
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256,256,3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return 0.006 * K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def l1_perceptual_loss(y_true, y_pred):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256,256,3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return 0.006*K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))+K.mean(K.abs(y_pred - y_true), axis=-1)
