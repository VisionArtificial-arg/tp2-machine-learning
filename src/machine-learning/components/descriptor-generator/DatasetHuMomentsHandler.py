import cv2 
import csv 
import glob 
import numpy 
import math 
from basic_image_processor.components.image_converter.grayscale_conversion import GrayscaleConversion
from basic_image_processor.components.threshold import AdaptiveGaussThreshold 
from basic_image_processor.components.morphological_transformers import Erosion  


class DatasetHuMomentsHandler:
    
    def __init__(label, writer):
        self.label  = label,
        self.writer = writer


    # Logica para manejar la generacion de momento de hu 

    # Se encarga de escribir los momentos de hu, en el archivo 
    def writer_hu_moments(label, writer):
        files = glob.glob('./shapes/' + label + '/*')  # label recibe el nombre de la carpeta
        hu_moments = []
        for file in files: 
            hu_moments.append(hu_moments_on_file(file))

        for moments in hu_moments:
            flattened = mom.ravel()  # paso de un array de arrays a un array simple.
            row = numpy.append(flattened, label)  # le metes el flattened array y le agregas el label
            writer.writerow(row)  # Escribe una linea en el archivo.
            
    def generate_hu_moments_file():
        with open('generated-files/shapes-hu-moments.csv', 'w', newline='') as file:  # Se genera un archivo nuevo (W=Write)
        
        writer = csv.writer(file)
        # Ahora escribo los momentos de Hu de cada uno de las figuras. Con el string "rectangle...etc" busca en la carpeta donde estan cada una de las imagenes
        # generar los momentos de Hu y los escribe sobre este archivo. (LOS DE ENTRENAMIENTO).
        write_hu_moments("5-point-star", writer)
        write_hu_moments("rectangle", writer)
        write_hu_moments("triangle", writer)

    # Encargada de generar los momentos de Hu para las imagenes
    def hu_moments_of_file(filename):
        image = cv2.imread(filename)
        gray = GrayscaleConversion.apply(image)
        bin = AdaptiveGaussThreshold(gray);

        # Invert the image so the area of the UAV is filled with 1's. This is necessary since
        # cv::findContours describes the boundary of areas consisting of 1's.
        bin = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

        kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer

        # erosion 
        bin = Erosion.apply(); 
        # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
        bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

        contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos
        shape_contour = max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima

        # Calculate Moments
        moments = cv2.moments(shape_contour)  # momentos de inercia
        
        # Calculate Hu Moments
        huMoments = cv2.HuMoments(moments)  # momentos de Hu
        
        # Log scale hu moments
        for i in range(0, 7):
            huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i])) # Mapeo para agrandar la escala.
        return huMoments



