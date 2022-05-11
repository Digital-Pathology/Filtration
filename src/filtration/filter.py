import abc
import cv2
from numbers import Number
import numpy as np
import random
from PIL.Image import fromarray as PIL_Image_from_array
from histolab.tile import Tile

# TODO: Add docstrings to all classes, methods, and top of each file

# Filter.filter() returns True if the region passes the filter


class Filter(abc.ABC):
    """ A abstract base class for all Filters """

    def __call__(self, region) -> bool:
        """ the interface for Filter protocols """
        return self.filter(region)

    def __str__(self):
        """ represent the filter for logging, etc. """
        return f"<{self.__class__.__name__}: {vars(self)}>"

    @abc.abstractmethod
    def filter(self, region) -> bool:
        """ details of the behavior of the Filter """
        pass


class FilterBlackAndWhite(Filter):
    """ Filters out regions with too much whitespace """

    def __init__(self, filter_threshold=0.5, binarization_threshold=0.85, rgb_weights=[0.2989, 0.5870, 0.1140]):
        """
        __init__ Initialize FilterBlackAndWhite Object

        :param filter_threshold: Threshold at which image region passes black and white filter, defaults to 0.5
        :type filter_threshold: float, optional
        :param binarization_threshold: Threshold to determine will be converted to either black or white, defaults to 0.85
        :type binarization_threshold: float, optional
        :param rgb_weights: Weighting used for RGB to greyscale conversion, defaults to [0.2989, 0.5870, 0.1140]
        :type rgb_weights: list, optional
        """
        self.filter_threshold = filter_threshold
        self.binarization_threshold = binarization_threshold
        self.rgb_weights = rgb_weights

    def filter(self, region) -> bool:
        """
        filter Perform filtration to a region

        :param region: numpy array representing the region
        :type region: np.ndarray
        :return: True if the average of the binary region is less than the filter threshold, else False
        :rtype: bool
        """
        greyscale_image = self.convert_rgb_to_greyscale(region)
        # if pixel is > 85% white, set value to 1 else 0
        binary_image = np.where(
            greyscale_image > self.binarization_threshold * 255, 1, 0)
        return bool(np.mean(binary_image) < self.filter_threshold)

    def convert_rgb_to_greyscale(self, region):
        """
        convert_rgb_to_greyscale Convert an RGB region of image to greyscale

        :param region: numpy array representing the region, region consists of RGB values
        :type region: np.ndarray
        :return: a numpy array representing the region, region is in greyscale
        :rtype: np.ndarray
        """
        return np.uint8(np.dot(region[..., :3], self.rgb_weights))


class FilterHSV(Filter):
    """ Filters out regions according to average hue """

    def __init__(self, threshold: Number = 100) -> None:
        """
        __init__ Initialize FilterHSV Object

        :param threshold: Threshold at which image region passes HSV filter, defaults to 100
        :type threshold: Number, optional
        """
        self.threshold = threshold

    def filter(self, region) -> bool:
        """
        filter Perform filtration to a region

        :param region: numpy array representing the region
        :type region: np.ndarray
        :return: True if the mean of the hues in the hsv region is greather than the threshold, else False
        :rtype: bool
        """
        hsv_img = self.convert_rgb_to_hsv(region)
        hue = hsv_img[:, :, 0]
        return bool(np.mean(hue) > self.threshold)

    def convert_rgb_to_hsv(self, region) -> np.ndarray:
        """
        convert_rgb_to_hsv Converts RGB region of image to HSV

        :param region: numpy array representing the region, region consists of RGB values
        :type region: np.ndarray
        :return: a numpy array representing the region, region consists of HSV values
        :rtype: np.ndarray
        """
        return cv2.cvtColor(region, cv2.COLOR_RGB2HSV)


class FilterFocusMeasure(Filter):
    """ Filters out regions that are sufficiently out-of-focus """

    def __init__(self, threshold=65.0) -> None:
        """
        __init__ Initialize filter for focus measure

        :param threshold: Threshold at which image region passes FocusMeasure filter, defaults to 65
        :type threshold: float, optional
        """
        self.threshold = threshold

    def filter(self, region) -> bool:
        """
        filter Perform filtration to a region by determining whether it is focused or blurry

        :param region: numpy array representing the region
        :type region: np.ndarray
        :return: True if the focus measure is greater than the supplied threshold (image is not
                considered blurry), else False (image is considered blurry)
        :rtype: bool
        """
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        focus_measure = self.variance_of_laplacian(gray)

        # if the focus measure is less than the supplied threshold, then the image is considered blurry
        if focus_measure < self.threshold:
            return False

        # show the image
        # cv2.putText(region, "{}: {:.2f}".format(text, focus_measure), (10, 30),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # cv2.imshow("Image", region)
        # key = cv2.waitKey(0)

        return True

    def variance_of_laplacian(self, region) -> float:
        """
        variance_of_laplacian Computes the Laplacian of the region

        :param region: numpy array representing the region
        :type region: np.ndarray
        :return: The focus measure which is the variance of the Laplacian
        :rtype: float
        """
        return cv2.Laplacian(region, cv2.CV_64F).var()


class FilterRandom(Filter):
    """ Decides whether a region passes randomly """

    def __init__(self, p: Number = 0.5) -> None:
        """
        __init__ initialize random filter

        :param p: threshold at which to filter randomly, defaults to 0.5
        :type p: Number, optional
        """
        self.p = p

    def filter(self, region) -> bool:
        """
        filter filters based on a random number being greater than the set threshold

        :param region: unused region of an image
        :type region: np.ndarray
        :return: Returns a boolean depending on whether RNG picks a number greater than the set threshold
        :rtype: bool
        """
        return random.random() > self.p


class FilterHistolab(Filter):
    """ Applies basic tissue filtration from the Histolab PyPi package """

    def __init__(self, tissue_percentage: float = 0.8) -> None:
        """
        __init__ Initialize FilterHistolab Object

        :param tissue_percentage: the percentage of tissue above which regions pass the filter, default 0.8
        :type filter_threshold: float, optional
        """
        self.tissue_percentage = tissue_percentage

    def filter(self, region) -> bool:
        """
        filter filters regions based on the histolab package's Tile.has_enough_tissue()

        :param region: the region to which filtration will be applied
        :type region: np.ndarray
        """
        region_tile = Tile(
            image=PIL_Image_from_array(region),
            coords=(0, 0)
        )
        return region_tile.has_enough_tissue(tissue_percent=self.tissue_percentage)
