import os
from kerassurgeon import utils
from keras import Model
import keras.backend as K
from keras.initializers import glorot_uniform
from keras.models import clone_model

def create_path(*parts):
    path = os.getcwd()
    for part in parts:
        path = os.path.join(path, part)
        if "." not in part and not os.path.isdir(path):
            os.makedirs(path)
    return path


def save_results_to_bucket():
    script_path = create_path('gsutil_to_bucket.sh')
    os.system('sh ' + script_path)


def get_activations(model, layer, x_val):
    act_layer, act_index = utils.find_activation_layer(layer, 0)

    temp_model = Model(inputs=model.inputs, outputs=act_layer.get_output_at(act_index))
    return temp_model.predict(x_val)


def reinitialise_weights(seed, discard_indices, model):
    all_weights = model.get_weights()
    fc_layer_biases = all_weights[-3]
    fc_layer_weights = all_weights[-4]

    k_eval = lambda x: x.eval(session=K.get_session())

    for i in discard_indices:
        fc_layer_biases[i] = k_eval(glorot_uniform(seed)([1]))[0]
        fc_layer_weights[..., i] = k_eval(glorot_uniform(seed)(fc_layer_weights[..., i].shape))

    target_model = clone_model(model)
    target_model.set_weights(all_weights)

    return target_model
