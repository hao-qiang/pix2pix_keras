import os
import numpy as np
import models
from tqdm import tqdm
import cv2
import scipy.misc


def img_preprocess(img_dir):
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (256,256))
    img = img[:,:,::-1] / 127.5 - 1
    return img


testset_dir = ''
save_dir = ''

generator_model = models.load("generator", (256, 256, 3))
generator_model.load_weights('./weights/pix2pix/gen_weights_epoch10.h5')

img_names = os.listdir(testset_dir)
num = len(img_names)
print(num)
input_img = np.zeros((1, 256, 256, 3))
for i in tqdm(range(num)):
    input_img[0] = img_preprocess(os.path.join(testset_dir, img_names[i]))
    output_img = generator_model.predict(input_img)
    scipy.misc.imsave(os.path.join(save_dir, img_names[i]), output_img[0])
