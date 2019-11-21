from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_activation, visualize_cam
from keras.models import load_model
from keras import activations
from matplotlib import pyplot as plt
import numpy as np

import input_data as data
from run import Run
from utils import create_path


def saliency_map(run, model_type, seed, dataset_type, attempt, category=None, positive=True):
    if not category:
        category = run.hyperparameters.target_animal

    img = _get_image(category, dataset_type, run, positive=positive)

    model = load_model(create_path(run.path, model_type, "saved_models", "{}_model_{}.h5".format(model_type, seed)))

    # layer_idx = utils.find_layer_idx(model, 'dense_1')
    layer_idx = 15
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    plt.imsave(create_path(run.path, model_type, "images", category, "{}_original_image.png".format(attempt)), img)

    grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=img[np.newaxis, ...])
    plt.imsave(
        create_path(run.path, model_type, "images", category, "sal", "{}_sal_image_seed{}.png".format(attempt, seed)),
        grads)

    grads_guided = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=img[np.newaxis, ...],
                                      backprop_modifier="guided")
    plt.imsave(create_path(run.path, model_type, "images", category, "sal",
                           "{}_sal_guided_image_seed{}.png".format(attempt, seed)), grads_guided)

    acti = visualize_activation(model, layer_idx, filter_indices=0, seed_input=img[np.newaxis, ...])
    plt.imsave(
        create_path(run.path, model_type, "images", category, "acti", "{}_acti_image_seed{}.png".format(attempt, seed)),
        acti)

    cam = visualize_cam(model, layer_idx, filter_indices=0, seed_input=img[np.newaxis, ...])
    plt.imsave(
        create_path(run.path, model_type, "images", category, "cam", "{}_cam_image_seed{}.png".format(attempt, seed)),
        cam)

    cam_guided = visualize_cam(model, layer_idx, filter_indices=0, seed_input=img[np.newaxis, ...],
                               backprop_modifier="guided")
    plt.imsave(create_path(run.path, model_type, "images", category, "cam",
                           "{}_cam_guided_image_seed{}.png".format(attempt, seed)), cam_guided)


def _get_image(category, dataset_type, run, positive=True):
    if dataset_type == "train":
        labels, images, test_labels, test_images = data.get_training_and_val_data(category, labels_per_category=run.
                                                                                  hyperparameters.labels_per_category)
    elif dataset_type == "test":
        train_labels, train_images, labels, images = data.get_training_and_val_data(category, labels_per_category=run.
                                                                                    hyperparameters.labels_per_category)
    else:
        labels, images = data.get_test_data(category)

    if positive:
        label = 1
    else:
        label = 0

    for index, image in enumerate(images):
        if labels[index] == label:
            img = image

    return img


def _alt_sal_map(model, layer_idx, img, run, model_type, seed):
    import keras.backend as K
    # select class of interest
    class_idx = 0
    # define derivative d loss / d layer_input
    layer_input = model.input
    # This model must already use linear activation for the final layer
    loss = model.layers[layer_idx].output[..., class_idx]
    grad_tensor = K.gradients(loss, layer_input)[0]

    # create function that evaluate the gradient for a given input
    # This function accept numpy array
    derivative_fn = K.function([layer_input], [grad_tensor])

    # evaluate the derivative_fn
    grad_eval_by_hand = derivative_fn([img[np.newaxis, ...]])[0]
    print(grad_eval_by_hand.shape)

    grad_eval_by_hand = np.abs(grad_eval_by_hand).max(axis=(0, 3))

    # normalize to range between 0 and 1
    arr_min, arr_max = np.min(grad_eval_by_hand), np.max(grad_eval_by_hand)
    grad_eval_by_hand = (grad_eval_by_hand - arr_min) / (arr_max - arr_min + K.epsilon())

    plt.imsave(create_path(run.path, model_type, "images", "sal_image_{}_alt.png".format(seed)), grad_eval_by_hand)


if __name__ == "__main__":
    r = Run.restore("exp_1", 1, 3, save_opp=True)
    for i in range(0, 3):
        for network in ['target', 'naive', 'source']:
            saliency_map(r, network, i, "train", 0, category="cat", positive=True)
            print("{} network done for seed {}".format(network, i))

    # for i in range(0, 3):
    #     saliency_map(r, 'target', i, "train", 0, category="dog", positive=True)
