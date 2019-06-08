import os
from kerassurgeon import utils
from keras import Model
import keras.backend as K
from keras.initializers import glorot_uniform

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


def reinitialise_weights(seed, discard_indices, layer):
    # TODO: use discard indicies to loop over neurons in layer and only change the input and output weights for those.

    old_input_weights = layer.input
    old_output_weights = layer.output

    k_eval = lambda x: x.eval(session=K.get_session())

    new_input_weights = [k_eval(glorot_uniform(seed)(w.shape)) for w in old_input_weights]
    new_output_weights = [k_eval(glorot_uniform(seed)(w.shape)) for w in old_output_weights]



    pass