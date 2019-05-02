import os
from kerassurgeon import utils
from keras import Model

def create_path(*parts):
    # path = os.path.dirname(__file__)
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
