"""Changes to libraries."""

from keras import Model, Sequential
from keras.callbacks import Callback, EarlyStopping
from kerassurgeon import utils
from kerassurgeon.surgeon import Surgeon
from keras import backend as k
import numpy as np
from sklearn.metrics import matthews_corrcoef


# Library bug fix.
def get_apoz(model, layer, x_val, node_indices=None):
    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    mean_calculator = utils.MeanCalculator(sum_axis=0)
    for node_index in node_indices:
        act_layer, act_index = utils.find_activation_layer(layer, node_index)
        # Get activations
        if hasattr(x_val, "__iter__"):
            temp_model = Model(model.inputs,
                               act_layer.get_output_at(act_index))
            a = temp_model.predict(x_val)
        else:
            get_activations = k.function(
                [utils.single_element(model.inputs), k.learning_phase()],
                [act_layer.get_output_at(act_index)])
            a = get_activations([x_val, 0])[0]
            # Ensure that the channels axis is last
        if data_format == 'channels_first':
            a = np.swapaxes(a, 1, -1)
        # Flatten all except channels axis
        activations = np.reshape(a, [-1, a.shape[-1]])
        zeros = (activations == 0).astype(int)
        mean_calculator.add(zeros)

    return mean_calculator.calculate()


# Records data at each epoch of training.
class PredictionHistory(Callback):
    def __init__(self, generate_mcc_results, generate_accuracy_results, model, train_images, train_labels, val_images,
                 val_labels, test_images, test_labels):
        super().__init__()
        self.generate_mcc_results = generate_mcc_results
        self.generate_accuracy_results = generate_accuracy_results
        self.model = model
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.train = {}
        self.val = {}
        self.test = {}

        if self.generate_mcc_results:
            self.train['mcc'] = []
            self.val['mcc'] = []
            self.test['mcc'] = []
            self.train['acc'] = []
            self.val['acc'] = []
            self.test['acc'] = []

    def on_epoch_begin(self, epoch, logs={}):
        if self.generate_mcc_results:
            pred_train_classes = self.model.predict_classes(self.train_images)
            self.train['mcc'].append(matthews_corrcoef(self.train_labels, pred_train_classes))

            pred_val_classes = self.model.predict_classes(self.val_images)
            self.val['mcc'].append(matthews_corrcoef(self.val_labels, pred_val_classes))

            pred_test_classes = self.model.predict_classes(self.test_images)
            self.test['mcc'].append(matthews_corrcoef(self.test_labels, pred_test_classes))

        if self.generate_accuracy_results:
            current_train_accuracy = self.model.evaluate(self.train_images, self.train_labels)[1]
            self.train['acc'].append(current_train_accuracy)

            current_val_accuracy = self.model.evaluate(self.val_images, self.val_labels)[1]
            self.val['acc'].append(current_val_accuracy)

            current_test_accuracy = self.model.evaluate(self.test_images, self.test_labels)[1]
            self.test['acc'].append(current_test_accuracy)

    def on_train_end(self, logs={}):
        if self.generate_mcc_results:
            pred_train_classes = self.model.predict_classes(self.train_images)
            self.train['mcc'].append(matthews_corrcoef(self.train_labels, pred_train_classes))

            pred_val_classes = self.model.predict_classes(self.val_images)
            self.val['mcc'].append(matthews_corrcoef(self.val_labels, pred_val_classes))

            pred_test_classes = self.model.predict_classes(self.test_images)
            self.test['mcc'].append(matthews_corrcoef(self.test_labels, pred_test_classes))

        if self.generate_accuracy_results:
            current_train_accuracy = self.model.evaluate(self.train_images, self.train_labels)[1]
            self.train['acc'].append(current_train_accuracy)

            current_val_accuracy = self.model.evaluate(self.val_images, self.val_labels)[1]
            self.val['acc'].append(current_val_accuracy)

            current_test_accuracy = self.model.evaluate(self.test_images, self.test_labels)[1]
            self.test['acc'].append(current_test_accuracy)

# Stopping the source network at the target accuracy peak.
class EarlyStoppingWithMax(EarlyStopping):
    def __init__(self, target=None, **kwargs):
        self.target = target
        super(EarlyStoppingWithMax, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if self.target and self.monitor_op(current, self.target):
            self.stopped_epoch = epoch
            self.model.stop_training = True

# Library bug fix.
class MySurgeon(Surgeon):
    def __init__(self, model):
        super(MySurgeon, self).__init__(model, copy=None)

    def operate(self):
        """Perform all jobs assigned to the surgeon.
        """
        # Operate on each node in self.nodes by order of decreasing depth.
        sorted_nodes = sorted(self.nodes, reverse=True,
                              key=lambda x: utils.get_node_depth(self.model, x))
        for node in sorted_nodes:
            # Rebuild submodel up to this node
            sub_output_nodes = utils.get_node_inbound_nodes(node)
            outputs, output_masks = self._rebuild_graph(self.model.inputs,
                                                        sub_output_nodes)

            # Perform surgery at this node
            kwargs = self._kwargs_map[node]
            self._mod_func_map[node](node, outputs, output_masks, **kwargs)

        # Finish rebuilding model
        output_nodes = []
        for output in self.model.outputs:
            layer, node_index, tensor_index = output._keras_history
            output_nodes.append(utils.get_inbound_nodes(layer)[node_index])
        new_outputs, _ = self._rebuild_graph(self.model.inputs, output_nodes)
        new_model = Model(self.model.inputs, new_outputs)
        new_model = Sequential(layers=new_model.layers)

        if self._copy:
            return utils.clean_copy(new_model)
        else:
            return new_model