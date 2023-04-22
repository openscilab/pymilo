from collections import namedtuple

import numpy as np

DistributionBoundary = namedtuple("DistributionBoundary", ("value", "inclusive"))
class CustomizedTweedieDistribution():

    def __init__(self, power=0):
        self.power = power

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        self._lower_bound = DistributionBoundary(0, inclusive=True)
        self._power = power

    def unit_variance(self, y_pred):
        return np.power(y_pred, self.power)

    def unit_deviance(self, y, y_pred, check_input=False):
        return (y - y_pred) ** 2

