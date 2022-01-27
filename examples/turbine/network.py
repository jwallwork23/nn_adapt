from nn_adapt.layout import NetLayoutBase


class NetLayout(NetLayoutBase):
    """
    Input layer:
    ============
        [mesh Reynolds number]
          + [drag coefficient]
          + [bathymetry]
          + [element size]
          + [element orientation]
          + [element shape]
          + [12 forward DoFs per element]
          + [12 adjoint DoFs per element]
          = 30

    Hidden layer:
    =============
        60 neurons

    Output layer:
    =============
        [1 error indicator value]
    """
    num_inputs = 30
    num_hidden_neurons = 60
    num_outputs = 1
