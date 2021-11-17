from solving import *


def get_hessians(f, **kwargs):
    """
    Compute Hessians for each component of
    a :class:`Function`.

    Any keyword arguments are passed to
    ``recover_hessian``.

    :arg f: the function
    :return: a dictionary containing the
        nested structure of the Hessian
    """
    return {[
            hessian_metric(recover_hessian(fij, **kwargs))
            for j, fij in enumerate(fi)
        ] for i, fi in split_into_scalars(f).items()
    }
