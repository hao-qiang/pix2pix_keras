import numpy as np
import cv2
import scipy.misc


def normalization(img):
    # rescale input img within [-1,1]
    return img / 127.5 - 1


def inverse_normalization(img):
    # rescale output img within [0,1], then saving by 'scipy.misc.imsave'
    return (img + 1.) / 2.


def read_one_img(img_dir):
    img = cv2.imread(img_dir)[:, :, ::-1]
    img = normalization(img)
    img_HR = img[:, 256:, :]
    img_LR = img[:, :256, :]
    return img_HR, img_LR


def gen_batch(X_list, batch_size=32):
    idx = np.random.choice(X_list.shape[0], batch_size, replace=False)
    X_HR_batch = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
    X_LR_batch = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)

    for i in range(batch_size):
        X_HR_batch[i], X_LR_batch[i] = read_one_img(X_list[idx[i]])
    return X_HR_batch, X_LR_batch


def get_disc_batch(X_HR_batch, X_LR_batch, G_model, batch_counter):
    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = G_model.predict(X_LR_batch)
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
        y_disc[:, 0] = 0
    else:
        X_disc = X_HR_batch
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
        y_disc[:, 0] = 1
    return X_disc, y_disc


def plot_generated_batch(X_HR, X_LR, G_model, epoch):
    # Generate images
    X_SR = G_model.predict(X_LR[:4])
    X_SR = inverse_normalization(X_SR)
    X_LR = inverse_normalization(X_LR[:4])
    X_HR = inverse_normalization(X_HR[:4])
    X = np.concatenate((X_LR, X_SR, X_HR), axis=0)

    list_rows = []
    for i in range(int(X.shape[0] // 4)):
        Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
        list_rows.append(Xr)

    Xr = np.concatenate(list_rows, axis=0)
    scipy.misc.imsave("./figures/val_epoch%s.png" % epoch, Xr)
