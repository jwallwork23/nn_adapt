__all__ = ["ConvergenceTracker"]


class ConvergenceTracker:
    """
    Class for checking convergence of fixed point
    iteration loops.
    """

    def __init__(self, mesh, parsed_args):
        self.qoi_old = None
        self.elements_old = mesh.num_cells()
        self.estimator_old = None
        self.converged_reason = None
        self.qoi_rtol = parsed_args.qoi_rtol
        self.element_rtol = parsed_args.element_rtol
        self.estimator_rtol = parsed_args.estimator_rtol
        self.fp_iteration = 0
        self.miniter = parsed_args.miniter
        self.maxiter = parsed_args.maxiter
        assert self.maxiter >= self.miniter

    def check_maxiter(self):
        """
        Check for reaching maximum number of iterations.
        """
        converged = False
        if self.fp_iteration >= self.maxiter:
            self.converged_reason = "reaching maximum iteration count"
            converged = True
        return converged

    def _chk(self, val, old, rtol, reason):
        converged = False
        if old is not None and self.fp_iteration >= self.miniter:
            if abs(val - old) < rtol * abs(old):
                self.converged_reason = reason
                converged = True
        return converged

    def check_qoi(self, val):
        """
        Check for QoI convergence.
        """
        r = "QoI convergence"
        converged = self._chk(val, self.qoi_old, self.qoi_rtol, r)
        self.qoi_old = val
        return converged

    def check_estimator(self, val):
        """
        Check for error estimator convergence.
        """
        r = "error estimator convergence"
        converged = self._chk(val, self.estimator_old, self.estimator_rtol, r)
        self.estimator_old = val
        return converged

    def check_elements(self, val):
        """
        Check for mesh element count convergence.
        """
        r = "element count convergence"
        converged = self._chk(val, self.elements_old, self.element_rtol, r)
        self.elements_old = val
        return converged
