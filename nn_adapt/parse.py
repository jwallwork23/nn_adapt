import argparse
import numpy as np


__all__ = ["Parser"]


def _check_in_range(value, typ, l, u):
    tvalue = typ(value)
    if not (tvalue >= l and tvalue <= u):
        raise argparse.ArgumentTypeError(f"{value} is not in [{l}, {u}]")
    return tvalue


def _check_strictly_in_range(value, typ, l, u):
    tvalue = typ(value)
    if not (tvalue >= l and tvalue <= u):
        raise argparse.ArgumentTypeError(f"{value} is not in ({l}, {u})")
    return tvalue


nonnegative_float = lambda value: _check_in_range(value, float, 0, np.inf)
nonnegative_int = lambda value: _check_in_range(value, int, 0, np.inf)
positive_float = lambda value: _check_strictly_in_range(value, float, 0, np.inf)
positive_int = lambda value: _check_strictly_in_range(value, int, 0, np.inf)


def bounded_float(l, u):
    def chk(value):
        return _check_in_range(value, float, l, u)

    return chk


def bounded_int(l, u):
    def chk(value):
        return _check_in_range(value, int, l, u)

    return chk


class Parser(argparse.ArgumentParser):
    """
    Custom :class:`ArgumentParser` for `nn_adapt`.
    """

    def __init__(self, prog):
        super().__init__(
            self, prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.add_argument("model", help="The model", type=str, choices=["turbine"])
        self.add_argument("test_case", help="The configuration file number or name")
        self.add_argument(
            "--optimise",
            help="Turn off plotting and debugging",
            action="store_true",
        )

    def parse_convergence_criteria(self):
        self.add_argument(
            "--miniter",
            help="Minimum number of iterations",
            type=positive_int,
            default=3,
        )
        self.add_argument(
            "--maxiter",
            help="Maximum number of iterations",
            type=positive_int,
            default=35,
        )
        self.add_argument(
            "--qoi_rtol",
            help="Relative tolerance for QoI",
            type=positive_float,
            default=0.001,
        )
        self.add_argument(
            "--element_rtol",
            help="Element count tolerance",
            type=positive_float,
            default=0.001,
        )
        self.add_argument(
            "--estimator_rtol",
            help="Error estimator tolerance",
            type=positive_float,
            default=0.001,
        )

    def parse_num_refinements(self, default=4):
        self.add_argument(
            "--num_refinements",
            help="Number of mesh refinements",
            type=positive_int,
            default=default,
        )

    def parse_approach(self):
        self.add_argument(
            "-a",
            "--approach",
            help="Adaptive approach to consider",
            choices=["isotropic", "anisotropic"],
            default="anisotropic",
        )
        self.add_argument(
            "--transfer",
            help="Transfer the solution from the previous mesh as initial guess",
            action="store_true",
        )

    def parse_target_complexity(self):
        self.add_argument(
            "--base_complexity",
            help="Base metric complexity",
            type=positive_float,
            default=200.0,
        )
        self.add_argument(
            "--target_complexity",
            help="Target metric complexity",
            type=positive_float,
            default=4000.0,
        )

    def parse_preproc(self):
        self.add_argument(
            "--preproc",
            help="Data preprocess function",
            type=str,
            choices=["none", "arctan", "tanh", "logabs"],
            default="arctan",
        )

    def parse_tag(self):
        self.add_argument(
            "--tag",
            help="Model tag (defaults to current git commit sha)",
            default=None,
        )
