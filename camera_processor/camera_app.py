import cv2

import pipeline

cv2.namedWindow("Controls")
cv2.createTrackbar("Threshold", "Controls", 127, 255, lambda x: None)
cv2.createTrackbar("Kernel", "Controls", 3, 20, lambda x: None)
cv2.createTrackbar("Iterations", "Controls", 1, 10, lambda x: None)

pipeline.manual_value = cv2.getTrackbarPos("Threshold", "Controls")
pipeline.kernel_size = cv2.getTrackbarPos("Kernel", "Controls") | 1  # asegurar impar
pipeline.iterations = cv2.getTrackbarPos("Iterations", "Controls")

processed = pipeline.process(frame)