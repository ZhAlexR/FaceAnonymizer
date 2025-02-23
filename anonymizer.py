import os.path
from typing import NamedTuple

import cv2
from cv2.typing import MatLike
from mediapipe.python.solutions.face_detection import FaceDetection


class FaceAnonymizer:

    def __init__(self, image_path: str, blur_weight: int = 50):
        self.image_path: str = image_path
        self._blur_kernel_size: tuple[int, int] = (blur_weight, ) * 2
        self._image: MatLike = cv2.imread(image_path)
        self._face_detector: FaceDetection = FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self._image_processed = False

    def blur_faces(self):
        faces = self._detect_faces().detections

        if faces is None:
            return

        for face in faces:
            x1, y1, x2, y2 = self._get_face_border_box_coordinates(face)
            self._blur_face((x1, y1, x2, y2))

        self._image_processed = True

    def save_image(self, image_name: str = None):

        if self._image_processed is False:
            print("Sorry, but you have to execute 'blur_faces' first")
            return

        split_path = self.image_path.split("/")
        image_path = os.path.join(*split_path[:-1])
        current_image_name_with_extension = split_path[-1]
        current_image_name, image_extension = current_image_name_with_extension.split(".")

        if image_name is None:
            image_name = f"{current_image_name}_blured"

        full_path = f"{image_path}/{image_name}.{image_extension}"
        cv2.imwrite(full_path, self._image)
        return f"{image_name}.{image_extension}"

    @property
    def image_rgb(self):
        return cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)

    def _detect_faces(self):
        with self._face_detector as detector:
            return detector.process(self.image_rgb)

    def _get_face_border_box_coordinates(self, face: NamedTuple):
        bbox = face.location_data.relative_bounding_box
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.width, bbox.height
        img_height, img_width, _ = self._image.shape

        return int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_width)

    def _blur_face(self, face_coordinates: tuple[int, int, int, int]):
        x1, y1, x2, y2 = face_coordinates
        self._image[y1: y1 + y2, x1: x1 + x2, :] = cv2.blur(self._image[y1: y1 + y2, x1: x1 + x2, :], self._blur_kernel_size)


def anonymize_file(file_name: str):
    path_to_file = os.path.join("user_temp_files", file_name)
    face_anonymizer = FaceAnonymizer(image_path=path_to_file)

    face_anonymizer.blur_faces()
    return face_anonymizer.save_image()
