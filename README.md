# pix2pix implemented by keras

This is a keras implementation of paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (pix2pix). I learned a lot from tdeboissiere's [code](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix).

---

## Requirements

- python 3.6
- keras 2.1.5
- tensorFlow 1.4
- opencv 3.4.1

---

## Getting Started

### 1. Preparing your data

All your training data and validation data should be processed into A2B form as shown in figure. Then moving the training images (3 channels) into folder `./data/train` , and validation images into folder `./data/val` .

![]()

### 2. Training

```
python train.py
```

- Model pictures and generated validation pictures will be saved in folder `./figures`, and weights will be saved in folder `./weights/pix2pix`.

- If you want to use perceptual loss, please replace `loss = [l1_loss, 'binary_crossentropy']` (line44) by `loss = [perceptual_loss, 'binary_crossentropy']` (line45) in `train.py`.

### 3. Testing

You need to change the path of folder `testset_dir` and `save_dir` , and run

```
python test.py
```

---

## Have fun!
