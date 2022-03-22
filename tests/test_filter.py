from pathlib import Path
from PIL import Image
import cv2
import pytest
from numpy import asarray
from src.filtration.filter_manager import FilterManager
from src.filtration.filter import Filter, FilterBlackAndWhite, FilterHSV, FilterFocusMeasure


TEST_MEDICAL_DIR = 'tests/test_medical_images'
TEST_GENERIC_DIR = 'tests/test_generic_images'


def test_filter_manager():
    black_and_white_filter: Filter = FilterBlackAndWhite()
    hsv_filter: Filter = FilterHSV()
    focus_measure_filter: Filter = FilterFocusMeasure()
    filter_manager = FilterManager(
        [black_and_white_filter, hsv_filter, focus_measure_filter])

    for path in Path(TEST_MEDICAL_DIR).rglob("*.*"):
        absolute_path = str(path.absolute())
        img = cv2.imread(absolute_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if 'pass' in absolute_path:
            assert filter_manager.filter(img)
        elif 'fail' in absolute_path:
            assert not filter_manager.filter(img)


@pytest.mark.parametrize("test_image, filter_threshold", [
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.4),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.5),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.6),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.7),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.7),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.8),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.9),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.07),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.1),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.2),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.3),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.9),
])
def test_filter_threshold_black_and_white_filter_pass(test_image, filter_threshold):
    black_white_filter: Filter = FilterBlackAndWhite(
        filter_threshold=filter_threshold)
    img = Image.open(test_image)
    assert black_white_filter(asarray(img))


@pytest.mark.parametrize("test_image, filter_threshold", [
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.05),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.1),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.2),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.3),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.2),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.5),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.6),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.00001),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.00009),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.0001),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.0003),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.0007),
])
def test_filter_threshold_black_and_white_filter_fail(test_image, filter_threshold):
    black_white_filter: Filter = FilterBlackAndWhite(
        filter_threshold=filter_threshold)
    img = Image.open(test_image)
    assert not black_white_filter(asarray(img))


@pytest.mark.parametrize("test_image, binarization_threshold", [
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.9),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.95),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 1.0),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 1.0),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 1.2),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 1.5),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 1.8),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.7),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.8),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.9),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 1.0),
])
def test_binarization_threshold_black_and_white_filter_pass(test_image, binarization_threshold):
    black_white_filter: Filter = FilterBlackAndWhite(
        binarization_threshold=binarization_threshold)
    img = Image.open(test_image)
    assert black_white_filter(asarray(img))


@pytest.mark.parametrize("test_image, binarization_threshold", [
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.05),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.2),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 0.8),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.05),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.3),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.7),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 0.01),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.05),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.1),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.3),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 0.6),
])
def test_binarization_threshold_black_and_white_filter_fail(test_image, binarization_threshold):
    black_white_filter: Filter = FilterBlackAndWhite(
        binarization_threshold=binarization_threshold)
    img = Image.open(test_image)
    assert not black_white_filter(asarray(img))


@pytest.mark.parametrize("test_image, threshold", [
    (f'{TEST_GENERIC_DIR}/cat.jpg', 10),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 30),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 50),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 63),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 70),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 75),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 10),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 30),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 50),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 10),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 30),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 50),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 63),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 70),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 75),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 90),
])
def test_hsv_filter_pass(test_image, threshold):
    hsv_filter: Filter = FilterHSV(threshold)
    img = Image.open(test_image)
    assert hsv_filter(asarray(img))


@pytest.mark.parametrize("test_image, threshold", [
    (f'{TEST_GENERIC_DIR}/cat.jpg', 90),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 100),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 130),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 150),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 200),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 60),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 90),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 150),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 100),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 120),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 150),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 200),
])
def test_hsv_filter_fail(test_image, threshold):
    hsv_filter: Filter = FilterHSV(threshold)
    img = Image.open(test_image)
    assert not hsv_filter(asarray(img))


@pytest.mark.parametrize("test_image, threshold", [
    (f'{TEST_GENERIC_DIR}/cat.jpg', 1),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 3),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 5),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 6),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 7),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 10),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 10),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 30),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 50),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 100),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 110),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 1),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 3),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 5),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 6),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 7),
])
def test_focus_measure_filter_pass(test_image, threshold):
    focus_measure_filter: Filter = FilterFocusMeasure(threshold)
    img = Image.open(test_image)
    assert focus_measure_filter(asarray(img))


@pytest.mark.parametrize("test_image, threshold", [
    (f'{TEST_GENERIC_DIR}/cat.jpg', 15),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 23),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 37),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 65),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 87),
    (f'{TEST_GENERIC_DIR}/cat.jpg', 100),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 120),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 130),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 150),
    (f'{TEST_GENERIC_DIR}/flamingo.jpg', 200),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 8),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 11),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 30),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 50),
    (f'{TEST_GENERIC_DIR}/dolphin.jpg', 100),
])
def test_focus_measure_filter_fail(test_image, threshold):
    focus_measure_filter: Filter = FilterFocusMeasure(threshold)
    img = Image.open(test_image)
    assert not focus_measure_filter(asarray(img))
