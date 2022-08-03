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

    inputs = None
    num_hidden_neurons_1 = None
    num_hidden_neurons_2 = None
    num_hidden_neurons_3 = None
    # TODO: Allow more general networks

    colours = {
        "estimator": "b",
        "physics": "C0",
        "mesh": "deepskyblue",
        "forward": "mediumturquoise",
        "adjoint": "mediumseagreen",
    }

    def __init__(self):
        if self.inputs is None:
            raise ValueError("Need to set inputs")
        colours = set(self.colours.keys())
        for i in self.inputs:
            okay = False
            for c in colours:
                if i.startswith(c):
                    okay = True
                    break
            if not okay:
                raise ValueError("Input names must begin with one of {colours}")
        if self.num_hidden_neurons_1 is None:
            raise ValueError("Need to set number of first hidden layer neurons ")
        if self.num_hidden_neurons_2 is None:
            raise ValueError("Need to set number of second hidden layer neurons")
        if self.num_hidden_neurons_3 is None:
            raise ValueError("Need to set number of third hidden layer neurons")

    def count_inputs(self, prefix):
        """
        Count all scalar inputs that start with a given `prefix`.
        """
        cnt = 0
        for i in self.inputs:
            if i.startswith(prefix):
                if i in ("forward_dofs", "adjoint_dofs"):
                    cnt += 12
                else:
                    cnt += 1
        return cnt

    @property
    def num_inputs(self):
        """
        The total number of scalar inputs.
        """
        return self.count_inputs("")
