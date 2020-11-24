import os
import pickle
import re, glob
import tensorflow as tf
import tensorflow_datasets as tfds

import configs
import utils

AUTOTUNE = tf.data.experimental.AUTOTUNE
CIFAR_LABELS = ['background', 'airplane', 'automobile', 'bird', 'cat', 'deer',
       'dog', 'frog', 'horse', 'ship', 'truck']

BRAIN_LABELS = ["background", "CSF", "gray matter", "white matter",
        "deep gray matter", "brain stem", "cerebellum"]

def load_data(dataset_name):
    # load data from tfds
    
    # Optionally split training data 
    if configs.config_values.split[0] != "100":
        split = configs.config_values.split
        split = [
            "train[:{}%]".format(split[0]),
            "train[-{}%:]".format(split[1]),
            "test"
        ]
    else:
        split = ["train","test"]
    print("Split:", split)

    if dataset_name in ["masked_fashion", "blown_fashion", "blown_masked_fashion"]:
        dataset_name="fashion_mnist"

    if dataset_name == "multiscale_cifar10":
        dataset_name = "cifar10"

    if dataset_name in ["masked_cifar10", "seg_cifar10"]:
        with open("data/masked_cifar10/masked_cifar10_strict.p", "rb") as f:
            data = pickle.load(f)
            train = tf.data.Dataset.from_tensor_slices(data)
        with open("data/masked_cifar10/masked_cifar10_strict_test.p", "rb") as f:
            data = pickle.load(f)
            test = tf.data.Dataset.from_tensor_slices(data)
        return train, test
    
    if "brain" in dataset_name:
        return load_brain_data()

    if "circles" == dataset_name:
        return load_circles()

    if "pet" in dataset_name:
        dataset = tfds.load('oxford_iiit_pet', data_dir="data")
        train, test = dataset["train"], dataset["test"]
        train = train.concatenate(test.shuffle(4000, seed=2020, reshuffle_each_iteration=False).take(3000))
        test  = test.skip(3000)
    else:
        data_generators = tfds.load(
            name=dataset_name, batch_size=-1,
            # data_dir="data",
            shuffle_files=False,
            split=split)
        
        # First and last will always be train/test
        # Potentially split could include a tune set which will be used later for learning the density
        # and will be ignored by score matching 
        train = tf.data.Dataset.from_tensor_slices(data_generators[0]['image'])
        test = tf.data.Dataset.from_tensor_slices(data_generators[-1]['image'])

    return train, test

def load_circles():
    with open("data/circles/train_smooth_64x64.p", "rb") as f:
        data = pickle.load(f)
        train = tf.data.Dataset.from_tensor_slices(data)
    
    with open("data/circles/test_smooth_64x64.p", "rb") as f:
        data = pickle.load(f)
        test = tf.data.Dataset.from_tensor_slices(data)

    return train, test


def load_brain_data():
    
    # Change this to whatever directory you use
    DATA_DIR = "./data/processed/images/" 
    train_paths = glob.glob(DATA_DIR+"/train/*")
    test_paths = glob.glob(DATA_DIR+"/test/*")
    
    # Create a dictionary describing the features.
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'segmentation': tf.io.FixedLenFeature([], tf.string),
    }

    @tf.function
    def _parse_record(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    @tf.function
    def _parse_mri(example_proto):
        
        # Get the record and put it into a feature dict
        image_features = _parse_record(example_proto)
        
        # Deserialize the mri array 
        mri = tf.io.parse_tensor(image_features['image'],
                                out_type=tf.float32)
        mask = tf.io.parse_tensor(image_features['mask'],
                                out_type=tf.float32)
        seg = tf.io.parse_tensor(image_features['segmentation'],
                                out_type=tf.float32)
        
        x = tf.concat((mri,mask,seg), axis=-1)
        return x

    train = tf.data.TFRecordDataset(train_paths).map(_parse_mri)
    test  = tf.data.TFRecordDataset(test_paths).map(_parse_mri)

    return train, test

@tf.function
def concat_mask(x):
    mask = tf.cast(x>0, dtype=tf.float32)
    return tf.concat((x,mask), axis=-1)

# @tf.autograph.experimental.do_not_convert
@tf.function
def pad(x):
    offset_h = tf.random.uniform([1], minval=0, maxval=28, dtype=tf.dtypes.int32)[0]
    offset_w = tf.random.uniform([1], minval=0, maxval=28, dtype=tf.dtypes.int32)[0]
    x =  tf.image.pad_to_bounding_box(x, offset_height=offset_h, offset_width=offset_w,
                                     target_height=56, target_width=56)
    return x

@tf.function
def preproc_pet_masks(x):
    img = tf.image.resize(x["image"], (64,64)) / 255 # resize + rescale [0,255] -> [0,1]
    mask = tf.image.resize(x["segmentation_mask"], (64,64))
    mask = tf.cast(mask != 2, dtype=tf.float32) # 1= background, 2,3=Foreground
    
    return tf.concat((img,mask), axis=-1)

@tf.function
def preproc_cifar_masks(x):
    img, mask = tf.split(x, (3,1), axis=-1)
    img  = img / 255
    mask = tf.cast(mask > 0, dtype=tf.float32) # 0= background, 1=Foreground
    
    return tf.concat((img,mask), axis=-1)

@tf.function
def preproc_cifar_segs(x):
    img, seg = tf.split(x, (3,1), axis=-1)
    img  = img / 255
    seg = tf.one_hot(tf.squeeze(seg), depth=11) # 0 = background, >0 = CIFAR label
    
    return tf.concat((img,seg), axis=-1) # Shape = 32x32x(3+11)

@tf.function
def preproc_cifar_multiscale(x):
    x = x/255
    x_small_scale = tf.image.resize(x, (8,8), method="bilinear")
    x_small_scale = tf.image.resize(x_small_scale, (32,32), method="nearest")
   
    return tf.concat((x,x_small_scale), axis=-1) # Shape = 32x32x(3+3)

@tf.function
def get_brain_only(x):
    img, mask,seg = tf.split(x, 3, axis=-1)
    img  = tf.expand_dims(img, axis=-1)
    return img

@tf.function
def get_brain_masks(x):
    img, mask,seg = tf.split(x, 3, axis=-1)
    x = tf.stack((img,mask), axis=-1)
    return x

@tf.function
def get_brain_segs(x):
    img, mask, seg = tf.split(x, 3, axis=-1)
    img  = tf.expand_dims(img, axis=-1)
    seg = tf.cast(tf.squeeze(seg), dtype=tf.int32)
    seg = tf.one_hot(seg, depth=7)
    x = tf.concat((img,seg), axis=-1)
    return x

preproc_map = {
    "brain": get_brain_only,
    "masked_brain": get_brain_masks,
    "seg_brain": get_brain_segs,
    "seg_cifar10": preproc_cifar_segs,
    "masked_cifar10": preproc_cifar_masks,
    "multiscale_cifar10": preproc_cifar_multiscale,
    "masked_pet": preproc_pet_masks,
}

def preprocess(dataset_name, data, train=True):

    if dataset_name in preproc_map:
        _fn = preproc_map[dataset_name]
        data = data.map(_fn, num_parallel_calls=AUTOTUNE)

    elif dataset_name not in ["masked_pet", "masked_cifar10"]:
        data = data.map(lambda x: x / 255, num_parallel_calls=AUTOTUNE)  # rescale [0,255] -> [0,1]
    
    if dataset_name == "pet":
        data = data.map(lambda x: tf.image.resize(x["image"], (64,64)))

    # Caching offline data
    data = data.cache()

    # Online augmentation 
    if dataset_name in ["blown_fashion", "blown_masked_fashion"]:
        data = data.map(pad)

    if dataset_name in ["masked_fashion", "blown_masked_fashion"]:
        data = data.map(concat_mask)
    
    if train and dataset_name in ["multiscale_cifar10", "cifar10", "masked_cifar10", "seg_cifar10", "pet", "masked_pet"]:
        data = data.map(lambda x: tf.image.random_flip_left_right(x),
                        num_parallel_calls=AUTOTUNE)  # randomly flip along the vertical axis

    if train and "brain" in dataset_name:
        data = data.map(lambda x: tf.image.random_flip_up_down(x), # randomly flip along the direction of hemispheres
                        num_parallel_calls=AUTOTUNE) 
    return data


def _preprocess_celeb_a(data, random_flip=True):
    # Discard labels and landmarks
    data = data.map(lambda x: x['image'], num_parallel_calls=AUTOTUNE)
    # Take a 140x140 centre crop of the image
    data = data.map(lambda x: tf.image.resize_with_crop_or_pad(x, 140, 140), num_parallel_calls=AUTOTUNE)
    # Resize to 32x32
    data = data.map(lambda x: tf.image.resize(x, (32, 32)), num_parallel_calls=AUTOTUNE)
    # Rescale
    data = data.map(lambda x: x / 255, num_parallel_calls=AUTOTUNE)
    # Maybe cache in memory
    # data = data.cache()
    # Randomly flip
    if random_flip:
        data = data.map(lambda x: tf.image.random_flip_left_right(x), num_parallel_calls=AUTOTUNE)
    return data


def get_celeb_a(random_flip=True):
    batch_size = configs.config_values.batch_size
    data_generators = tfds.load(name='celeb_a', batch_size=batch_size, data_dir="data", shuffle_files=True)
    train = data_generators['train']
    test = data_generators['test']
    train = _preprocess_celeb_a(train, random_flip=random_flip)
    test = _preprocess_celeb_a(test, random_flip=False)
    return train, test


def get_celeb_a32():
    """
    Loads the preprocessed celeb_a dataset scaled down to 32x32
    :return: tf.data.Dataset with single batch as big as the whole dataset
    """
    path = './data/celeb_a32'
    if not os.path.exists(path):
        print(path, " does not exits")
        return None

    images = utils.get_tensor_images_from_path(path)
    data = tf.data.Dataset.from_tensor_slices(images)
    data = data.map(lambda x: tf.cast(x, tf.float32))
    data = data.batch(int(tf.data.experimental.cardinality(data)))
    return data

def get_ood_data(dataset_name):
    print("Getting OOD Dataset...")
    OOD_LABEL = 0
    data = tfds.load(name="mnist", batch_size=-1, data_dir="data", shuffle_files=False)
    
    mask = data["train"]["label"] != OOD_LABEL
    inlier_train = tf.data.Dataset.from_tensor_slices(data["train"]["image"][mask])
    
    mask = data["test"]["label"] != OOD_LABEL
    inlier_test = tf.data.Dataset.from_tensor_slices(data["test"]["image"][mask])

    mask = data["test"]["label"] == OOD_LABEL
    ood_test = tf.data.Dataset.from_tensor_slices(data["test"]["image"][mask])

    inlier_train = preprocess("mnist", inlier_train, train=True)
    inlier_test = preprocess("mnist", inlier_test, train=False)
    ood_test = preprocess("mnist", ood_test, train=False)

    return inlier_train, inlier_test, ood_test

def get_train_test_data(dataset_name):

    if dataset_name == 'mnist_ood':
        train,test,_ = get_ood_data(dataset_name)
    elif dataset_name != 'celeb_a':
        train, test = load_data(dataset_name)
        train = preprocess(dataset_name, train, train=True)
        test = preprocess(dataset_name, test, train=False)
    else:
        train, test = get_celeb_a()
    return train, test


def get_data_inpainting(dataset_name, n):
    if dataset_name == 'celeb_a':
        data = get_celeb_a(random_flip=False)[0]
        data = next(iter(data.take(1)))[:n]
    else:
        data_generator = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", split='train', shuffle_files=True)
        data = data_generator['image']
        data = tf.random.shuffle(data, seed=1000)
        data = data[:n] / 255
    return data


def get_data_k_nearest(dataset_name):
    data_generator = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", split='train', shuffle_files=False)
    data = tf.data.Dataset.from_tensor_slices(data_generator['image'])
    data = data.map(lambda x: tf.cast(x, dtype=tf.float32))

    return data
