import os.path
from typing import NamedTuple

import cv2
from cv2.typing import MatLike
from mediapipe.python.solutions.face_detection import FaceDetection


DEFAULT_FPS = 25
DEFAULT_FILE_FOLDER = "./user_temp_files/"

class FaceAnonymizer:

    def __init__(self, face_detector: FaceDetection, blur_weight: int = 50):
        self._blur_kernel_size: tuple[int, int] = (blur_weight, ) * 2
        self._image: MatLike | None = None
        self._face_detector: FaceDetection = face_detector
        self._image_processed = False

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image: MatLike):
        self._image = image

    @property
    def image_rgb(self) -> MatLike:
        return cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)


    def blur_faces(self) -> MatLike:

        self._check_image_exists()

        faces = self._detect_faces().detections

        if faces is None:
            return

        for face in faces:
            x1, y1, x2, y2 = self._get_face_border_box_coordinates(face)
            self._blur_face((x1, y1, x2, y2))

        return self._image

    def _check_image_exists(self):
        if not self.image:
            raise ValueError(
                "You have to provide an image before blurring!\n"
                "Initialize FaceAnonymizer `instance` and use `instance.image = image`"
            )

    def _detect_faces(self) -> NamedTuple:
        return self._face_detector.process(self.image_rgb)

    def _get_face_border_box_coordinates(self, face: NamedTuple) -> tuple[int, int, int, int]:
        bbox = face.location_data.relative_bounding_box
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.width, bbox.height
        img_height, img_width, _ = self._image.shape

        return int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_width)

    def _blur_face(self, face_coordinates: tuple[int, int, int, int]):
        x1, y1, x2, y2 = face_coordinates
        self._image[y1: y1 + y2, x1: x1 + x2, :] = cv2.blur(self._image[y1: y1 + y2, x1: x1 + x2, :], self._blur_kernel_size)


def blur_and_save_image(face_anonymizer: FaceAnonymizer, path_to_image: str) -> str:
    image = cv2.imread(path_to_image)
    face_anonymizer.image = image
    image = face_anonymizer.blur_faces()
    cv2.imwrite(path_to_image, image)
    return path_to_image

def blur_and_save_video(face_anonymizer: FaceAnonymizer, path_to_file: str) -> str:
    video = cv2.VideoCapture(path_to_file)
    ret, frame = video.read()

    output_file_path = os.path.join(DEFAULT_FILE_FOLDER, "output.mp4")
    output_video = cv2.VideoWriter(
        output_file_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        DEFAULT_FPS,
        (frame.shape[1], frame.shape[0])
    )

    while ret:
        face_anonymizer.image = frame
        frame = face_anonymizer.blur_faces()
        output_video.write(frame)

        ret, frame = video.read()

    video.release()
    output_video.release()
    return output_file_path


def anonymize_file(file_name: str, file_type: str):
    path_to_file = os.path.join(DEFAULT_FILE_FOLDER, file_name)
    face_detector = FaceDetection(model_selection=0, min_detection_confidence=0.7)
    face_anonymizer = FaceAnonymizer(face_detector=face_detector, blur_weight=100)

    if file_type == "photo":
        return blur_and_save_image(face_anonymizer, path_to_file)

    if file_type == "video":
        return blur_and_save_video(face_anonymizer, path_to_file)

    face_detector.close()


if __name__ == "__main__":
    anonymize_file("alina_video.mp4", "video")