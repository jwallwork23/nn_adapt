import abc


class Parameters(abc.ABC):
    """
    Abstract base class defining the API for parameter
    classes that describe PDE models.
    """

    def __init__(self):
        self.case = None
        if not hasattr(self, "qoi_name"):
            raise NotImplementedError("qoi_name attribute must be set")
        if not hasattr(self, "qoi_unit"):
            raise NotImplementedError("qoi_unit attribute must be set")

    @abc.abstractmethod
    def bathymetry(self, mesh):
        """
        Compute the bathymetry field on the current `mesh`.
        """
        pass

    @abc.abstractmethod
    def drag(self, mesh):
        """
        Compute the drag coefficient on the current `mesh`.
        """
        pass

    @abc.abstractmethod
    def viscosity(self, mesh):
        """
        Compute the viscosity coefficient on the current `mesh`.
        """
        pass

    @abc.abstractmethod
    def ic(self, mesh):
        """
        Compute the initial condition on the current `mesh`.
        """
        pass
