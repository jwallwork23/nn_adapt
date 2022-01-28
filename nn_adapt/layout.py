"""
Classes for defining the layout of a neural network.
"""


class NetLayoutBase(object):
    """
    Base class for specifying the number
    of inputs, hidden neurons and outputs
    in a neural network.

    The derived class should give values
    for each of these parameters.
    """
    num_inputs = None
    num_hidden_neurons = None
    num_outputs = None

    def __init__(self):
        if self.num_inputs is None:
            raise ValueError('Need to set number of inputs')
        if self.num_hidden_neurons is None:
            raise ValueError('Need to set number of hidden neurons')
        if self.num_outputs is None:
            raise ValueError('Need to set number of outputs')
