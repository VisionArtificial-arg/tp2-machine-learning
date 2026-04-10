import cv2

from components.image_pipeline import ImageProcessingPipeline
from components.inference.artist import InferenceArtist
from components.inference.inference_runner import InferenceRunner


def main():

    pipeline = ImageProcessingPipeline()

    cv2.namedWindow("Controls")
    cv2.createTrackbar("Threshold", "Controls", 127, 255, lambda x: None)
    cv2.createTrackbar("Kernel", "Controls", 3, 20, lambda x: None)
    cv2.createTrackbar("Iterations", "Controls", 1, 10, lambda x: None)

    cap = cv2.VideoCapture(0)
    runner = InferenceRunner(
        model_path="generated_files/model.joblib",
        testing_images_path="./shapes/testing"
    )
    artist = InferenceArtist()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pipeline.manual_value = cv2.getTrackbarPos("Threshold", "Controls")
        pipeline.kernel_size = cv2.getTrackbarPos("Kernel", "Controls") | 1
        pipeline.iterations = cv2.getTrackbarPos("Iterations", "Controls")

        processed = pipeline.process(frame)

        detections = runner.predict_many_from_frame(
            frame,
            min_area=700,
            max_area_ratio=0.40,
            ignore_border=True,
        )
        labels = [label for label, _ in detections]
        print("Labels:", labels)

        for label, contour in detections:
            frame = artist.draw(frame, contour, label)


        cv2.imshow("Camera", frame)
        cv2.imshow("Processed", processed)

        if cv2.waitKey(1) == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()