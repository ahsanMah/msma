#!/usr/bin/env python

import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("./")
DATADIR = "./data/"

import utils
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from ood_detection_helper import *
from datasets.dataset_loader import  *
from tqdm import tqdm

from PIL import Image
from IPython.display import display
from matplotlib.pyplot import imshow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


print(sys.argv)
SIGMA_HIGH = float(sys.argv[1])
NUM_L = int(sys.argv[2])
SAVE_NAME = "score_exploration/ablation/SH{:.0e}_L{:d}.p".format(SIGMA_HIGH, NUM_L)
MODELDIR = str(sys.argv[3])

if MODELDIR == "l":
    MODELDIR = "./longleaf_models/"
else:
    MODELDIR = "./saved_models/"

print(SAVE_NAME)

scores = {}
with open(SAVE_NAME, "wb") as f:
    pickle.dump(scores, f)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')


model = load_model(inlier_name="cifar10", s_high=SIGMA_HIGH,
                   num_L=NUM_L, save_path=MODELDIR)

BATCHSIZE = 1000


with tf.device('CPU'):
    data_generators = tfds.load(name="cifar10", batch_size=-1, shuffle_files=False)
    cifar10_test = tf.data.Dataset.from_tensor_slices(data_generators['test']["image"])
    cifar10_test = cifar10_test.map(lambda x: x/255, num_parallel_calls=AUTOTUNE)
    cifar10_test = cifar10_test.batch(BATCHSIZE).cache()
    
    cifar10_train  = tf.data.Dataset.from_tensor_slices(data_generators['train']["image"])
    cifar10_train = cifar10_train.map(lambda x: x/255, num_parallel_calls=AUTOTUNE)
    cifar10_train = cifar10_train.batch(BATCHSIZE).cache()

with tf.device('CPU'):
    data_generators = tfds.load(name="svhn_cropped", batch_size=-1, shuffle_files=False)
    svhn_test = tf.data.Dataset.from_tensor_slices(data_generators['test']["image"])
#     svhn_test = svhn_test.take(26000)
    svhn_test = svhn_test.map(lambda x: x/255, num_parallel_calls=AUTOTUNE)
    svhn_test = svhn_test.batch(BATCHSIZE).cache()


cifar10_train_scores = compute_weighted_scores(model, cifar10_train)
cifar10_scores = compute_weighted_scores(model, cifar10_test)
svhn_scores = compute_weighted_scores(model, svhn_test)

scores = {}
scores["train"] = cifar10_train_scores
scores["cifar"] = cifar10_scores
scores["svhn"] = svhn_scores


datasets = ["LSUN", "LSUN_resize", "Imagenet", "Imagenet_resize", "iSUN"] 
img_height, img_width = 32, 32

@tf.function
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

@tf.function
def process_path(file_path):
  # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img /= 255 
    return img


test_ds = []
BATCHSIZE = 512
N_BATCHES = 10
for ds_name in datasets:
    data_dir = DATADIR + ds_name
    list_ds = tf.data.Dataset.list_files(str(data_dir+'/*/*'))
    ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCHSIZE)
#     ds = ds.take(1)
    ds = ds.prefetch(buffer_size=AUTOTUNE).cache()
    test_ds.append(ds)

progress_bar = zip(datasets, test_ds)

for name, outlier in progress_bar:
    print(name)
    scores[name] = compute_weighted_scores(model, outlier)


x = tf.random.normal(shape=(BATCHSIZE*N_BATCHES, 32,32,3) , mean=0.5, stddev=1.0)
x = tf.clip_by_value(x, 0.0, 1.0)
gaussian_test_batches = tf.split(x, N_BATCHES)

x = tf.random.uniform(shape=(BATCHSIZE*N_BATCHES, 32,32,3), minval=0.0, maxval=1.0)
uniform_test_batches = tf.split(x, N_BATCHES)



artificial_ood = (gaussian_test_batches, uniform_test_batches)
for name, outlier in zip(["gaussian", "uniform"], artificial_ood):
    scores[name] = compute_weighted_scores(model,outlier)

with open(SAVE_NAME, "wb") as f:
    pickle.dump(scores, f)

