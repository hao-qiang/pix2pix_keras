from keras.models import Model
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization


def generator_model(img_dim, model_name="generator"):
    x_input = Input(shape=img_dim, name="input")
    x1_1 = Conv2D(64, (3, 3), strides=(2, 2), padding="same", name="1_conv")(x_input)
    x1_2 = LeakyReLU(0.2, name="1_lrelu")(x1_1)

    x2_1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same", name="2_conv")(x1_2)
    x2_2 = BatchNormalization(name="2_bn")(x2_1)
    x2_3 = LeakyReLU(0.2, name="2_lrelu")(x2_2)

    x3_1 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", name="3_conv")(x2_3)
    x3_2 = BatchNormalization(name="3_bn")(x3_1)
    x3_3 = LeakyReLU(0.2, name="3_lrelu")(x3_2)

    x4_1 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="4_conv")(x3_3)
    x4_2 = BatchNormalization(name="4_bn")(x4_1)
    x4_3 = LeakyReLU(0.2, name="4_lrelu")(x4_2)

    x5_1 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="5_conv")(x4_3)
    x5_2 = BatchNormalization(name="5_bn")(x5_1)
    x5_3 = LeakyReLU(0.2, name="5_lrelu")(x5_2)

    x6_1 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="6_conv")(x5_3)
    x6_2 = BatchNormalization(name="6_bn")(x6_1)
    x6_3 = LeakyReLU(0.2, name="6_lrelu")(x6_2)

    x7_1 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="7_conv")(x6_3)
    x7_2 = BatchNormalization(name="7_bn")(x7_1)
    x7_3 = LeakyReLU(0.2, name="7_lrelu")(x7_2)

    x8_1 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="8_conv")(x7_3)
    x8_2 = BatchNormalization(name="8_bn")(x8_1)
    x8_3 = Activation("relu", name="8_relu")(x8_2)

    x9_1 = UpSampling2D(size=(2, 2), name="9_upsample")(x8_3)
    x9_2 = Conv2D(512, (3, 3), padding="same", name="9_conv")(x9_1)
    x9_3 = BatchNormalization(name="9_bn")(x9_2)
    x9_4 = Dropout(0.5, name="9_dropout")(x9_3)
    x9_5 = Concatenate(name="9_concate")([x9_4, x7_2])
    x9_6 = Activation("relu", name="9_relu")(x9_5)

    x10_1 = UpSampling2D(size=(2, 2), name="10_upsample")(x9_6)
    x10_2 = Conv2D(512, (3, 3), padding="same", name="10_conv")(x10_1)
    x10_3 = BatchNormalization(name="10_bn")(x10_2)
    x10_4 = Dropout(0.5, name="10_dropout")(x10_3)
    x10_5 = Concatenate(name="10_concate")([x10_4, x6_2])
    x10_6 = Activation("relu", name="10_relu")(x10_5)

    x11_1 = UpSampling2D(size=(2, 2), name="11_upsample")(x10_6)
    x11_2 = Conv2D(512, (3, 3), padding="same", name="11_conv")(x11_1)
    x11_3 = BatchNormalization(name="11_bn")(x11_2)
    x11_4 = Dropout(0.5, name="11_dropout")(x11_3)
    x11_5 = Concatenate(name="11_concate")([x11_4, x5_2])
    x11_6 = Activation("relu", name="11_relu")(x11_5)

    x12_1 = UpSampling2D(size=(2, 2), name="12_upsample")(x11_6)
    x12_2 = Conv2D(512, (3, 3), padding="same", name="12_conv")(x12_1)
    x12_3 = BatchNormalization(name="12_bn")(x12_2)
    x12_4 = Concatenate(name="12_concate")([x12_3, x4_2])
    x12_5 = Activation("relu", name="12_relu")(x12_4)

    x13_1 = UpSampling2D(size=(2, 2), name="13_upsample")(x12_5)
    x13_2 = Conv2D(256, (3, 3), padding="same", name="13_conv")(x13_1)
    x13_3 = BatchNormalization(name="13_bn")(x13_2)
    x13_4 = Concatenate(name="13_concate")([x13_3, x3_2])
    x13_5 = Activation("relu", name="13_relu")(x13_4)

    x14_1 = UpSampling2D(size=(2, 2), name="14_upsample")(x13_5)
    x14_2 = Conv2D(128, (3, 3), padding="same", name="14_conv")(x14_1)
    x14_3 = BatchNormalization(name="14_bn")(x14_2)
    x14_4 = Concatenate(name="14_concate")([x14_3, x2_2])
    x14_5 = Activation("relu", name="14_relu")(x14_4)

    x15_1 = UpSampling2D(size=(2, 2), name="15_upsample")(x14_5)
    x15_2 = Conv2D(64, (3, 3), padding="same", name="15_conv")(x15_1)
    x15_3 = BatchNormalization(name="15_bn")(x15_2)
    x15_4 = Concatenate(name="15_concate")([x15_3, x1_1])
    x15_5 = Activation("relu", name="15_relu")(x15_4)

    x16_1 = UpSampling2D(size=(2, 2), name="16_upsample")(x15_5)
    x16_2 = Conv2D(3, (3, 3), padding="same", name="16_conv")(x16_1)
    x_output = Activation("tanh", name="output")(x16_2)  # keep pixel value in [-1,1]

    generator = Model(inputs=x_input, outputs=x_output, name=model_name)

    return generator


def discriminator_model(img_dim, model_name="discriminator"):
    x_input = Input(shape=img_dim, name="input")
    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", name="conv_1")(x_input)
    x = LeakyReLU(0.2, name="lrelu_1")(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", name="conv_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = LeakyReLU(0.2, name="lrelu_2")(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same", name="conv_3")(x)
    x = BatchNormalization(name="bn_3")(x)
    x = LeakyReLU(0.2, name="lrelu_3")(x)

    x = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv_4")(x)
    x = BatchNormalization(name="bn_4")(x)
    x = LeakyReLU(0.2, name="lrelu_4")(x)

    x = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv_5")(x)
    x = BatchNormalization(name="bn_5")(x)
    x = LeakyReLU(0.2, name="lrelu_5")(x)

    x = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv_6")(x)
    x = BatchNormalization(name="bn_6")(x)
    x = LeakyReLU(0.2, name="lrelu_6")(x)

    x = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv_7")(x)
    x = BatchNormalization(name="bn_7")(x)
    x = LeakyReLU(0.2, name="lrelu_7")(x)

    x = Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv_8")(x)
    x = BatchNormalization(name="bn_8")(x)
    x = LeakyReLU(0.2, name="lrelu_8")(x)

    x_flat = Flatten(name="flatten")(x)
    x_output = Dense(1, activation="sigmoid", name="output")(x_flat)

    discriminator = Model(inputs=x_input, outputs=x_output, name=model_name)

    return discriminator


def DCGAN(generator_model, discriminator_model, img_dim, model_name="DCGAN"):
    G_input = Input(shape=img_dim, name="input")
    G_output = generator_model(G_input)
    D_output = discriminator_model(G_output)
    DCGAN = Model(inputs=[G_input],
                  outputs=[G_output, D_output],
                  name=model_name)
    DCGAN.summary()
    from keras.utils import plot_model
    plot_model(DCGAN, to_file="./figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
    return DCGAN


def load(model_name, img_dim):
    if model_name == "generator":
        model = generator_model(img_dim, model_name=model_name)
        print("load generator")
        model.summary()
        from keras.utils import plot_model
        plot_model(model, to_file="./figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "discriminator":
        model = discriminator_model(img_dim, model_name=model_name)
        print("load discriminator")
        model.summary()
        from keras.utils import plot_model
        plot_model(model, to_file="./figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model


if __name__ == "__main__":
    load("generator", (256, 256, 3))
    load("discriminator", (256, 256, 3))
