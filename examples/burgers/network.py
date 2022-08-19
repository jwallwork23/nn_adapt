from nn_adapt.layout import NetLayoutBase


class NetLayout(NetLayoutBase):
    """
    Default configuration
    =====================

    Input layer:
    ------------
          [viscosity coefficient]
          + [element size]
          + [element orientation]
          + [element shape]
          + [boundary element?]
          + [12 forward DoFs per element]
          + [12 adjoint DoFs per element]
          = 29

    Hidden layer:
    -------------

        58 neurons

    Output layer:
    -------------

        [1 error indicator value]
    """

    inputs = (
        # "estimator_coarse",
        "physics_viscosity",
        "mesh_d",
        "mesh_h1",
        "mesh_h2",
        "mesh_bnd",
        "forward_dofs",
        "adjoint_dofs",
    )
    num_hidden_neurons = 58
    dofs_per_element = 12
