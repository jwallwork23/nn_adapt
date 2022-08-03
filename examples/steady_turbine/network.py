from nn_adapt.layout import NetLayoutBase


class NetLayout(NetLayoutBase):
    """
    Default configuration
    =====================

    Input layer:
    ------------
        [coarse-grained DWR]
          + [drag coefficient]
          + [viscosity coefficient]
          + [bathymetry]
          + [element size]
          + [element orientation]
          + [element shape]
          + [boundary element?]
          + [12 forward DoFs per element]
          + [12 adjoint DoFs per element]
          = 32

    Hidden layer 1:
    -------------

        25 neurons

    Hidden layer 2:
    -------------

        25 neurons

    Hidden layer 3:
    -------------

        26 neurons

    Output layer:
    -------------

        [1 error indicator value]
    """

    inputs = (
        "estimator_coarse",
        "physics_drag",
        "physics_viscosity",
        "physics_bathymetry",
        "mesh_d",
        "mesh_h1",
        "mesh_h2",
        "mesh_bnd",
        "forward_dofs",
        "adjoint_dofs",
    )
    num_hidden_neurons_1 = 25
    num_hidden_neurons_2 = 25
    num_hidden_neurons_3 = 26
