import math
import cv2
from cv2 import Mat

from basic_image_processor.components.image_converter.grayscale_conversion import GrayScaleConverter
from basic_image_processor.components.morphological_transformers import Erosion
from basic_image_processor.components.threshold import AdaptiveGaussThreshold


def preprocess_for_contours(image):
    gray = GrayScaleConverter().apply(image)
    binary = AdaptiveGaussThreshold().apply(gray)
    binary = 255 - binary

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = Erosion().apply(binary, kernel=kernel, iterations=1)
    return binary


def main_contour_of_image(image):
    binary = preprocess_for_contours(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def _is_border_contour(contour, width, height, margin=5):
    x, y, w, h = cv2.boundingRect(contour)
    return (
        x <= margin
        or y <= margin
        or (x + w) >= (width - margin)
        or (y + h) >= (height - margin)
    )


def contours_of_image(image, min_area=500, max_area_ratio=0.98, ignore_border=False):
    binary = preprocess_for_contours(image)
    h, w = binary.shape[:2]
    image_area = float(h * w)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if area > image_area * max_area_ratio:
            continue
        if ignore_border and _is_border_contour(contour, w, h):
            continue
        valid.append(contour)

    return sorted(valid, key=cv2.contourArea, reverse=True)

def compute_hu_moments(image):
    contour = main_contour_of_image(image)
    if contour is None:
        return None

    # momentos
    m = cv2.moments(contour)
    hu = cv2.HuMoments(m)

    # evitar log(0)
    eps = 1e-12
    for i in range(7):
        v = hu[i][0]
        if abs(v) < eps:
            hu[i][0] = 0.0
        else:
            hu[i][0] = -1 * math.copysign(1.0, v) * math.log10(abs(v))

    return hu


def contour_and_hu_moments_of_frame(frame):
    contour = main_contour_of_image(frame)
    if contour is None:
        return None, None
    m = cv2.moments(contour)
    hu = cv2.HuMoments(m)

    eps = 1e-12
    for i in range(7):
        v = hu[i][0]
        if abs(v) < eps:
            hu[i][0] = 0.0
        else:
            hu[i][0] = -1 * math.copysign(1.0, v) * math.log10(abs(v))

    return contour, hu


def contour_hu_pairs_of_frame(frame, min_area=500, max_area_ratio=0.98, ignore_border=False):
    pairs = []
    for contour in contours_of_image(
        frame,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        ignore_border=ignore_border,
    ):
        m = cv2.moments(contour)
        hu = cv2.HuMoments(m)

        eps = 1e-12
        for i in range(7):
            v = hu[i][0]
            if abs(v) < eps:
                hu[i][0] = 0.0
            else:
                hu[i][0] = -1 * math.copysign(1.0, v) * math.log10(abs(v))

        pairs.append((contour, hu))
    return pairs

def hu_moments_of_file(filename):
    return compute_hu_moments(cv2.imread(filename))

def hu_moments_of_frame(frame):
    return compute_hu_moments(frame)