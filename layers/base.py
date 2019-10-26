import tensorflow as tf
from utils.computil import ComputationContextLayer


def _get_regularizer_from_weight_decay(self, weight_decay):
    if isinstance(weight_decay, (list, tuple)):
        l1, l2 = weight_decay
        return tf.keras.regularizers.L1L2(l1=l1, l2=l2)
    else:
        return tf.keras.regularizers.L1L2(l1=0.0, l2=weight_decay)


class ComposeLayer(tf.keras.layers.Layer):
    """
    A layer that uses other layer as sublayer for assistant computation
    """
    def __init__(self, *args, **kwargs):
        super(ComposeLayer, self).__init__(*args, **kwargs)
        self.sub_layers = dict()
        self._c = None

    def add_layer(self, name, layer_type, *args, **kwargs):
        """
        Add a layer as its sublayer for computation if the unique name of the layer if not exists
        :param name: An unique name of that layer
        :param layer_type: The class of the layer
        :param args: The args for initialization of the layer
        :param kwargs: The kwargs for initialization of the layer
        :return: An instance of that layer
        """
        if self.sub_layers.get(name) is None:
            self.sub_layers[name] = ComputationContextLayer(layer_type(*args, **kwargs))
        return self.sub_layers[name]

    def dense(self, name, units, activation, weight_decay):
        """
        A simple alias to create a Dense layer in compose layer
        :param name: The name of the layer
        :param units: The number of output features of the final feature dimensions
        :param activation: The activation
        :param weight_decay: The weight decay, a float value for l2 weight decay, or a two tuple for l1,l2 weight decay
        :return: A dense layer
        """
        return self.add_layer(
            name,
            tf.keras.layers.Dense,
            units=units,
            activation=activation,
            kernel_regularizer=_get_regularizer_from_weight_decay(weight_decay),
            bias_regularizer=_get_regularizer_from_weight_decay(weight_decay),
            name=name
        )

    def batch_normalization(self, name, momentum, weight_decay):
        """
        Add a batch normalization layer in the compose layer
        :param name: The name of the layer
        :param momentum: The momentum of the batch normalization
        :param weight_decay: The weight decay for the layer
        :return: A batch normalization layer
        """
        return self.add_layer(
            name,
            tf.keras.layers.BatchNormalization,
            momentum=momentum,
            beta_regularizer=_get_regularizer_from_weight_decay(weight_decay),
            gamma_regularizer=_get_regularizer_from_weight_decay(weight_decay),
            name=name
        )

    def activation_(self, name, activation):
        """
        Add a activation layer in the compose layer
        :param name: The name of the layer
        :param activation: The activation of the layer
        :return: A activation layer
        """
        return self.add_layer(
            name,
            tf.keras.layers.Activation,
            activation=activation,
            name=name
        )

    def unary_convolution(self, name, channel, activation, momentum, weight_deacy):
        """
        Adding a unary convolution layer (in KPConv, Dense --> Normalization --> Activation).
        :param name: The name of the unary convolution
        :param channel: The output channel for the unary convolution
        :param activation: The activation, None for not using any activation
        :param momentum: The momentum used in batch normalization, none for not using any batch normalization
        :param weight_deacy
        :return: The unary convolution layer
        """
        ls = [self.dense(name + "-1x1 Convolution", channel, activation=None, weight_decay=weight_deacy)]
        if momentum is not None:
            ls.append(self.batch_normalization(name + "-Normalization", momentum=momentum, weight_deacy=weight_deacy))
        if activation is not None:
            ls.append(self.activation_(name, activation=activation))

        return tuple(ls)