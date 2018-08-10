import os
import numpy as np
import models
from keras.utils import generic_utils
from keras.optimizers import Adam
from data_utils import gen_batch, get_disc_batch, plot_generated_batch
from loss import l1_loss, perceptual_loss


batch_size = 32
n_batch_per_epoch = 200
nb_epoch = 40
model_name = "pix2pix"
if not os.path.exists("./models/" + model_name):
    os.makedirs("./models/" + model_name)
epoch_size = n_batch_per_epoch * batch_size

# Load and rescale data
train_dir = './data/train/'
val_dir = './data/val/'

train_list = [train_dir + i for i in os.listdir(train_dir)]
train_list = np.asarray(train_list)
val_list = [val_dir + i for i in os.listdir(val_dir)]
val_list = np.asarray(val_list)


# Create optimizers
G_opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
D_opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Load generator model
generator_model = models.load("generator", img_dim=(256, 256, 3))
# generator_model.load_weights('./models/pix2pix/gen_weights_epoch_6.h5')
generator_model.compile(loss='mae', optimizer=G_opt)

# Load discriminator model
discriminator_model = models.load("discriminator", img_dim=(256, 256, 3))
# discriminator_model.load_weights('./models/pix2pix/disc_weights_epoch_6.h5')
discriminator_model.trainable = False

DCGAN_model = models.DCGAN(generator_model, discriminator_model, img_dim=(256, 256, 3))
# DCGAN_model.load_weights('./models/pix2pix/DCGAN_weights_epoch_6.h5')

loss = [l1_loss, 'binary_crossentropy']
# loss = [perceptual_loss, 'binary_crossentropy']
loss_weights = [1E1, 1]
DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=G_opt)

discriminator_model.trainable = True
discriminator_model.compile(loss='binary_crossentropy', optimizer=D_opt)

# Start training
print("Start training")
for e in range(1, nb_epoch+1):
    # Initialize progbar and batch counter
    progbar = generic_utils.Progbar(epoch_size)
    print('Epoch %s/%s' % (e, nb_epoch))
    
    for b in range(1, n_batch_per_epoch+1):
        X_HR_batch, X_LR_batch = gen_batch(train_list, batch_size)
        # Create a batch to feed the discriminator model
        X_disc, y_disc = get_disc_batch(X_HR_batch, X_LR_batch, generator_model, b)

        # Update the discriminator
        disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

        # Create a batch to feed the generator model
        X_gen_target, X_gen = gen_batch(train_list, batch_size)
        y_gen = np.zeros((X_gen.shape[0], 1), dtype=np.uint8)
        y_gen[:, 0] = 1

        # Freeze the discriminator
        discriminator_model.trainable = False
        gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
        # Unfreeze the discriminator
        discriminator_model.trainable = True

        progbar.add(batch_size, values=[("D log_loss", disc_loss),
                                        ("G tot_loss", gen_loss[0]),
                                        ("G l1_loss", gen_loss[1]),
                                        ("G log_loss", gen_loss[2])])

        # Save images for visualization
        if b % (n_batch_per_epoch // 8) == 0:
            # Get new images from validation
            X_HR_batch, X_LR_batch = gen_batch(val_list, batch_size)
            plot_generated_batch(X_HR_batch, X_LR_batch, generator_model, e)

    print("")

    if e % 2 == 0:
        gen_weights_path = os.path.join('./weights/%s/gen_weights_epoch%s.h5' % (model_name, e))
        generator_model.save_weights(gen_weights_path, overwrite=True)

        # disc_weights_path = os.path.join('./models/%s/disc_weights_epoch%s.h5' % (model_name, e))
        # discriminator_model.save_weights(disc_weights_path, overwrite=True)

        # DCGAN_weights_path = os.path.join('./models/%s/DCGAN_weights_epoch%s.h5' % (model_name, e))
        # DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)
