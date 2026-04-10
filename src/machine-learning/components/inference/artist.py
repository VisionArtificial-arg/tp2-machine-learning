import cv2


class InferenceArtist:
    def __init__(self, contour_color=(0, 255, 0), text_color=(255, 255, 0)):
        self.contour_color = contour_color
        self.text_color = text_color

    def draw(self, image, contour, label):
        if contour is None or label is None:
            return image

        cv2.drawContours(image, [contour], -1, self.contour_color, 2)

        x, y, w, h = cv2.boundingRect(contour)
        text_position = (x, max(20, y - 10))

        cv2.putText(
            image,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.text_color,
            2,
            cv2.LINE_AA,
        )
        return image
