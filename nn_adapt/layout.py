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

    # TODO: Allow more general networks

    colours = {
        "estimator": "b",
        "physics": "C0",
        "mesh": "deepskyblue",
        "forward": "mediumturquoise",
        "adjoint": "mediumseagreen",
    }

    def __init__(self):
        if not hasattr(self, "inputs"):
            raise ValueError("Need to set self.inputs")
        colours = set(self.colours.keys())
        for i in self.inputs:
            okay = False
            for c in colours:
                if i.startswith(c):
                    okay = True
                    break
            if not okay:
                raise ValueError("Input names must begin with one of {colours}")
        if not hasattr(self, "num_hidden_neurons"):
            raise ValueError("Need to set self.num_hidden_neurons")
        if not hasattr(self, "dofs_per_element"):
            raise ValueError("Need to set self.dofs_per_element")

    def count_inputs(self, prefix):
        """
        Count all scalar inputs that start with a given `prefix`.
        """
        cnt = 0
        for i in self.inputs:
            if i.startswith(prefix):
                if i in ("forward_dofs", "adjoint_dofs"):
                    cnt += self.dofs_per_element
                else:
                    cnt += 1
        return cnt

    @property
    def num_inputs(self):
        """
        The total number of scalar inputs.
        """
        return self.count_inputs("")
