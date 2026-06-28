from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


class Transform:
    def __init__(self, dx=0.0, dy=0.0, angle=0.0):
        self.dx = dx
        self.dy = dy
        self.angle = angle


class ImagePairTransformer:
    def __init__(
        self,
        visible,
        infrared,
        root=REPO_ROOT,
        target_height=480,
        target_width=640,
        canvas_height=960,
        canvas_width=1280,
    ):
        self.root = root
        self.target_height = target_height
        self.target_width = target_width
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.visible = self._load_image(visible)
        self.infrared = self._load_image(infrared)

    def _resolve_path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _load_image(self, source):
        if isinstance(source, np.ndarray):
            return source.copy()

        path = self._resolve_path(source)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return image

    @staticmethod
    def resize_to_canvas(image, height, width):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def center_crop(image, height, width):
        image_height, image_width = image.shape[:2]
        if height > image_height or width > image_width:
            raise ValueError(f"Crop {(height, width)} is larger than image {(image_height, image_width)}")

        y1 = (image_height - height) // 2
        x1 = (image_width - width) // 2
        return image[y1:y1 + height, x1:x1 + width]

    @staticmethod
    def affine_transform(image, transform):
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0),
            transform.angle,
            1.0,
        )
        matrix[0, 2] += transform.dx
        matrix[1, 2] += transform.dy

        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def transform_image(
        self,
        image,
        transform=None,
        target_height=None,
        target_width=None,
        canvas_height=None,
        canvas_width=None,
    ):
        transform = transform or Transform()
        target_height = target_height or self.target_height
        target_width = target_width or self.target_width
        canvas_height = canvas_height or self.canvas_height
        canvas_width = canvas_width or self.canvas_width

        canvas = self.resize_to_canvas(image, canvas_height, canvas_width)
        transformed = self.affine_transform(canvas, transform)
        return self.center_crop(transformed, target_height, target_width)

    def transform_visible(self, transform=None):
        return self.transform_image(self.visible, transform)

    def transform_infrared(self, transform=None):
        return self.transform_image(self.infrared, transform)

    def transform_pair(self, visible=None, infrared=None):
        return {
            "visible": self.transform_visible(visible),
            "infrared": self.transform_infrared(infrared),
        }

    @staticmethod
    def save_image(image, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), image):
            raise ValueError(f"Could not save image: {output_path}")

        return output_path

    @staticmethod
    def save_pair(images, visible_path, infrared_path):
        return {
            "visible": ImagePairTransformer.save_image(images["visible"], visible_path),
            "infrared": ImagePairTransformer.save_image(images["infrared"], infrared_path),
        }


def main():
    transformer = ImagePairTransformer(
        "data_sources/lasher/testingset/boywalkinginsnow2/visible/000000.jpg",
        "data_sources/lasher/testingset/boywalkinginsnow2/infrared/000000.jpg",
    )

    transformed = transformer.transform_pair(
        visible=Transform(dx=-3, dy=-10),
        infrared=Transform(dx=5, dy=4),
    )

    print("visible:", transformed["visible"].shape)
    print("infrared:", transformed["infrared"].shape)

    saved_paths = transformer.save_pair(
        transformed,
        "outputs/transformed_pair/visible_shifted.jpg",
        "outputs/transformed_pair/infrared_shifted.jpg",
    )
    print("saved:", saved_paths)


if __name__ == "__main__":
    main()
