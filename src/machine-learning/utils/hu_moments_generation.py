import math
import cv2
from cv2 import Mat

from basic_image_processor.components.image_converter.grayscale_conversion import GrayScaleConverter
from basic_image_processor.components.morphological_transformers import Erosion
from basic_image_processor.components.threshold import AdaptiveGaussThreshold


def hu_moments_of_file(filename: str):
    image = cv2.imread(filename) # read image

    gray = GrayScaleConverter().apply(image) # applies grayscale

    binary = AdaptiveGaussThreshold().apply(gray) # apply adaptive threshold

    binary = 255 - binary # invert values

    # applies erosion
    # Aclaracion a futuro: podemos redefinir Erosion de nuestro basic_image_processor
    # esto para evitar tener que crear siempre el kernel, es decir, podemos setear valores por default en la creacion, y listo

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = Erosion().apply(binary, kernel=kernel, iterations=1)


    # find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    shape_contour = max(contours, key=cv2.contourArea)

    # get moments
    moments = cv2.moments(shape_contour)
    hu = cv2.HuMoments(moments)

    for i in range(7):
        v = hu[i][0]
        hu[i][0] = -1 * math.copysign(1.0, v) * math.log10(abs(v))

    return hu

def hu_moments_of_frame(frame: Mat):
    gray = GrayScaleConverter().apply(frame) # applies grayscale

    binary = AdaptiveGaussThreshold().apply(gray) # apply adaptive threshold

    binary = 255 - binary # invert values

    # applies erosion
    # Aclaracion a futuro: podemos redefinir Erosion de nuestro basic_image_processor
    # esto para evitar tener que crear siempre el kernel, es decir, podemos setear valores por default en la creacion, y listo

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = Erosion().apply(binary, kernel=kernel, iterations=1)


    # find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    shape_contour = max(contours, key=cv2.contourArea)

    # get moments
    moments = cv2.moments(shape_contour)
    hu = cv2.HuMoments(moments)

    for i in range(7):
        v = hu[i][0]
        hu[i][0] = -1 * math.copysign(1.0, v) * math.log10(abs(v))

    return hu
