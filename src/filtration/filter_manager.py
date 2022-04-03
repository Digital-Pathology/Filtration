from functools import reduce
from typing import Union

import numpy as np
from .filter import Filter


class FilterManager:

    def __init__(self, filters: Union[Filter, str, list]):
        """
        __init__ Initializes a filter manager that manages filters and applies them to input

        :param filters: Either a single Filter, the cls string of a filter, or a list of filters
        :type filters: Union[Filter, str, list]
        """

        self.filters = []
        if isinstance(filters, list):
            for filter in filters:
                self.add_filter(filter)
        else:
            self.add_filter(filters)

    def __call__(self, region):
        """
        __call__ Allows the filter manager to be act as a function

        :param region: Calls the filter function.
        :type region: np.ndarray
        """        
        self.filter(region)

    def __str__(self):
        """
        __str__ Returns all of the filters in self.filters as a list of strings

        :return: A list of strings representing the filters names
        :rtype: List[str]
        """        
        return str([str(f) for f in self.filters])

    def add_filter(self, filter: Union[Filter, str]):
        """
        add_filter Appends the input filter to the list of filters in state

        :param filter: Either a Filter subclass or a str representing the filter's type
        :type filter: Union[Filter, str]
        :raises Exception: This filter does not exist.
        :raises TypeError: This is not a filter.
        """        
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
        """
        filter Filters the region based on the input filters

        :param region: A region of the image to be filtered.
        :type region: np.ndarray
        :return: True if the region passes all of the filters, and false if not.
        :rtype: bool
        """        
        return reduce(lambda p, q: p and q, [filter.filter(region) for filter in self.filters])
