import os.path
import copy
from typing import NamedTuple

import cv2
from cv2.typing import MatLike
from mediapipe.python.solutions.face_detection import FaceDetection

DEFAULT_FPS = 25
DEFAULT_FILE_FOLDER = "user_temp_files"
VIDEO_OUTPUT_FILENAME = "output.mp4"
FOURCC_MP4 = "mp4v"


class FaceAnonymizer:
    """A class to anonymize faces in images or videos by blurring them."""

    def __init__(self, face_detector: FaceDetection, blur_weight: int = 50):
        self.face_detector = face_detector
        self._blur_kernel_size = (blur_weight, blur_weight)  # Consistent tuple notation

    def blur_faces(self, image: MatLike) -> MatLike:
        """Blur faces in the provided image and return the result.

        Args:
            image: The input image in BGR format.

        Returns:
            A copy of the image with faces blurred.

        Raises:
            ValueError: If the input image is None.
        """
        if image is None:
            raise ValueError("Input image cannot be None. Provide a valid image to blur.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.process(image_rgb).detections

        blurred_image = copy.copy(image)

        if faces:
            for face in faces:
                left_x, top_y, width, height = self._get_face_border_box_coordinates(face, image)
                blurred_image[top_y:top_y + height, left_x:left_x + width] = self._blur_face(
                    blurred_image[top_y:top_y + height, left_x:left_x + width]
                )

        return blurred_image

    @staticmethod
    def _get_face_border_box_coordinates(face: NamedTuple, image: MatLike) -> tuple[int, int, int, int]:
        """Extract face bounding box coordinates from detection data.

        Args:
            face: A NamedTuple containing face detection data.
            image: The image to calculate coordinates for.

        Returns:
            A tuple (left_x, top_y, width, height) of integer coordinates.
        """
        bbox = face.location_data.relative_bounding_box
        img_height, img_width, _ = image.shape
        left_x = int(bbox.xmin * img_width)
        top_y = int(bbox.ymin * img_height)
        width = int(bbox.width * img_width)
        height = int(bbox.height * img_height)
        return left_x, top_y, width, height

    def _blur_face(self, face_region: MatLike) -> MatLike:
        """Apply blur to a specific face region.

        Args:
            face_region: The image region containing the face.

        Returns:
            The blurred face region.
        """
        return cv2.blur(face_region, self._blur_kernel_size)



def blur_and_save_image(face_anonymizer: FaceAnonymizer, path_to_image: str) -> str:
    """Blur faces in an image and save it to the original path.

    Args:
        face_anonymizer: The FaceAnonymizer instance to use.
        path_to_image: Path to the input image file.

    Returns:
        The path where the blurred image was saved.

    Raises:
        FileNotFoundError: If the input image file doesn’t exist.
    """
    if not os.path.exists(path_to_image):
        raise FileNotFoundError(f"Image file not found: {path_to_image}")

    image = cv2.imread(path_to_image)
    if image is None:
        raise ValueError(f"Failed to load image from {path_to_image}")

    blurred_image = face_anonymizer.blur_faces(image)
    cv2.imwrite(path_to_image, blurred_image)
    return path_to_image


def blur_and_save_video(face_anonymizer: FaceAnonymizer, path_to_file: str) -> str:
    """Blur faces in a video and save it to a new file.

    Args:
        face_anonymizer: The FaceAnonymizer instance to use.
        path_to_file: Path to the input video file.

    Returns:
        The path where the blurred video was saved.

    Raises:
        FileNotFoundError: If the input video file doesn’t exist.
    """
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"Video file not found: {path_to_file}")

    video = cv2.VideoCapture(path_to_file)
    ret, frame = video.read()

    if not ret:
        raise ValueError(f"Failed to read video from {path_to_file}")

    output_file_path = os.path.join(".", DEFAULT_FILE_FOLDER, VIDEO_OUTPUT_FILENAME)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    video_writer = cv2.VideoWriter(
        output_file_path,
        cv2.VideoWriter_fourcc(*FOURCC_MP4),
        DEFAULT_FPS,
        (frame.shape[1], frame.shape[0])
    )

    while ret:
        blurred_frame = face_anonymizer.blur_faces(frame)
        video_writer.write(blurred_frame)
        ret, frame = video.read()


    video.release()
    video_writer.release()

    return output_file_path


def process_and_save_anonymized_file(file_name: str, file_type: str) -> str:
    """Process a file (image or video) by anonymizing faces and saving the result.

    Args:
        file_name: Name of the file in DEFAULT_FILE_FOLDER.
        file_type: Type of file ("photo" or "video").

    Returns:
        The path to the anonymized file.

    Raises:
        ValueError: If file_type is invalid.
        FileNotFoundError: If the file doesn’t exist (propagated from helpers).
    """
    path_to_file = os.path.join(".", DEFAULT_FILE_FOLDER, file_name)

    with FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        face_anonymizer = FaceAnonymizer(face_detector=face_detector, blur_weight=100)

        if file_type == "photo":
            return blur_and_save_image(face_anonymizer, path_to_file)
        elif file_type == "video":
            return blur_and_save_video(face_anonymizer, path_to_file)
        else:
            raise ValueError(f"Invalid file_type: {file_type}. Use 'photo' or 'video'.")


if __name__ == "__main__":
    try:
        result = process_and_save_anonymized_file("alina_video.mp4", "video")
        print(f"Processed file saved to: {result}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")