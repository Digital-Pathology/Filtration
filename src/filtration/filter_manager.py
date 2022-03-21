from functools import reduce
from typing import Union

import numpy as np
from .filter import Filter


class FilterManager:

    def __init__(self, filters: Union[Filter, str, list]):

        self.filters = []
        if isinstance(filters, list):
            for filter in filters:
                self.add_filter(filter)
        else:
            self.add_filter(filters)

    def __call__(self, region):
        self.filter(region)

    def __str__(self):
        return str([str(f) for f in self.filters])

    def add_filter(self, filter: Union[Filter, str]):
        filter_classes = {f.__name__: f for f in Filter.__subclasses__()}
        if isinstance(filter, Filter):
            pass
        elif isinstance(filter, str):  # string should match name of class
            if filter not in filter_classes:
                raise Exception(
                    f"filter {filter} does not exist! Currently available filters are: {list(filter_classes.keys())}"
                )
            else:
                filter = filter_classes[filter]()
        else:
            raise TypeError(type(filter))
        self.filters.append(filter)

    def filter(self, region: np.ndarray) -> bool:
        return reduce(lambda p, q: p and q,
                      [filter.filter(region) for filter in self.filters])
