from collections import defaultdict
from keras.datasets import cifar10
import numpy as np
from sklearn.utils import shuffle


# Data:
# The data, split between train and test sets:
__RAW_DATA__ = None


def _get_original_dataset():
    global __RAW_DATA__
    if __RAW_DATA__ is None:
        __RAW_DATA__ = cifar10.load_data()
    (images_train, labels_train), (images_test, labels_test) = __RAW_DATA__
    return labels_train, images_train, labels_test, images_test

def _get_category_by_name(category_name):
    if category_name == "plane":
        return 0
    elif category_name == "car":
        return 1
    elif category_name == "bird":
        return 2
    elif category_name == "cat":
        return 3
    elif category_name == "deer":
        return 4
    elif category_name == "dog":
        return 5
    elif category_name == "frog":
        return 6
    elif category_name == "horse":
        return 7
    elif category_name == "ship":
        return 8
    elif category_name == "truck":
        return 9
    else:
        print("No category with name {}".format(category_name))


# Balanced training data. Balanced validation data at 20% subset of training data
def get_training_and_val_data(animal):
    chosen_category = _get_category_by_name(animal)

    labels_train, images_train, labels_test, images_test = _get_original_dataset()
    filtered_labels, filtered_images = _balance_data(labels_train, images_train, 10, 5000, chosen_category)

    shuffled_filtered_labels, shuffled_filtered_images = shuffle(filtered_labels, filtered_images,
                                                                 random_state=0)

    if chosen_category == 0:
        shuffled_filtered_labels[shuffled_filtered_labels == 1] = 10
        shuffled_filtered_labels[shuffled_filtered_labels == chosen_category] = 1
        shuffled_filtered_labels[shuffled_filtered_labels != 1] = 0
    elif chosen_category == 1:
        shuffled_filtered_labels[shuffled_filtered_labels != chosen_category] = 0
    else:
        shuffled_filtered_labels[shuffled_filtered_labels != chosen_category] = 0
        shuffled_filtered_labels[shuffled_filtered_labels == chosen_category] = 1

    split_at = 7992
    (shuffled_filtered_images, val_images) = \
        shuffled_filtered_images[:split_at], shuffled_filtered_images[split_at:]
    (shuffled_filtered_labels, val_labels) = \
        shuffled_filtered_labels[:split_at], shuffled_filtered_labels[split_at:]

    shuffled_filtered_images = shuffled_filtered_images.astype('float32')
    shuffled_filtered_images /= 255
    val_images = val_images.astype('float32')
    val_images /= 255

    assert np.logical_or(shuffled_filtered_labels == 0, shuffled_filtered_labels == 1).all()
    assert np.logical_or(val_labels == 0, val_labels == 1).all()
    assert shuffled_filtered_labels.shape[0] == 7992
    assert val_labels.shape[0] == 1998

    return shuffled_filtered_labels, shuffled_filtered_images, val_labels, val_images


# Balanced test data
def get_test_data(animal):
    chosen_category = _get_category_by_name(animal)

    labels_train, images_train, labels_test, images_test = _get_original_dataset()
    filtered_labels, filtered_images = _balance_data(labels_test, images_test, 10, 1000, chosen_category)

    shuffled_filtered_labels, shuffled_filtered_images = shuffle(filtered_labels,
                                                                           filtered_images,
                                                                           random_state=0)

    if chosen_category == 0:
        shuffled_filtered_labels[shuffled_filtered_labels == 1] = 10
        shuffled_filtered_labels[shuffled_filtered_labels == chosen_category] = 1
        shuffled_filtered_labels[shuffled_filtered_labels != 1] = 0
    elif chosen_category == 1:
        shuffled_filtered_labels[shuffled_filtered_labels != chosen_category] = 0
    else:
        shuffled_filtered_labels[shuffled_filtered_labels != chosen_category] = 0
        shuffled_filtered_labels[shuffled_filtered_labels == chosen_category] = 1


    shuffled_filtered_images = shuffled_filtered_images.astype('float32')
    shuffled_filtered_images /= 255

    assert np.logical_or(shuffled_filtered_labels == 0, shuffled_filtered_labels == 1).all()
    assert shuffled_filtered_labels.shape[0] == 1998

    return shuffled_filtered_labels, shuffled_filtered_images


# Dogs imbalanced testing data
def get_imbal_labels(animal):
    chosen_category = _get_category_by_name(animal)

    labels_train, images_train, labels_test, images_test = _get_original_dataset()
    imbal_test_labels = np.empty(labels_test.shape)
    np.copyto(imbal_test_labels, labels_test)

    imbal_train_labels = np.empty(labels_train.shape)
    np.copyto(imbal_train_labels, labels_train)

    imbal_test_labels[imbal_test_labels != chosen_category] = 0
    imbal_test_labels[imbal_test_labels == chosen_category] = 1

    imbal_train_labels[imbal_train_labels != chosen_category] = 0
    imbal_train_labels[imbal_train_labels == chosen_category] = 1

    return imbal_train_labels, imbal_test_labels


# Input images formatting
def get_imbal_original_images():
    labels_train, images_train, labels_test, images_test = _get_original_dataset()
    images_train = images_train.astype('float32')
    images_train /= 255
    images_test = images_test.astype('float32')
    images_test /= 255

    return images_train, images_test


# Balances data 50% chosen category, even split of others.
def _balance_data(label_array, image_array, n_categories, labels_per_category, chosen_category):
    assert label_array.shape[0] == image_array.shape[0]

    labels = []
    images = []

    n_other_labels = n_categories - 1
    n_chosen = labels_per_category
    n_other = labels_per_category / n_other_labels

    while not n_other.is_integer():
        n_chosen -= 1
        n_other = n_chosen / n_other_labels

    c = defaultdict(int)

    for i, label in enumerate(label_array):
        l = label[0]
        if (l == chosen_category and c[l] < n_chosen) or (c[l] < n_other):
            labels.append(label)
            images.append(image_array[i])
            c[l] += 1
        else:
            continue

    return np.array(labels), np.array(images)


def get_category_images(label_array, image_array, chosen_category):
    images = []
    c = defaultdict(int)

    for i, label in enumerate(label_array):
        l = label[0]
        if (l == chosen_category):
            images.append(image_array[i])
            c[l] += 1
        else:
            continue

    return np.array(images)
