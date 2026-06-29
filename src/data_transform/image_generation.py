from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import randint
from tqdm import tqdm

import cv2
import json
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


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
        allowed_resolutions=None,
    ):
        self.root = root
        self.target_height = target_height
        self.target_width = target_width
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.allowed_resolutions = allowed_resolutions or []
        self.visible = self._load_image(visible)
        self.infrared = self._load_image(infrared)
        self.is_valid = self.visible is not None and self.infrared is not None

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
        if self.allowed_resolutions and image.shape[:2] not in self.allowed_resolutions:
            return None
        return image

    @staticmethod
    def resize_to_canvas(image, height, width):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def center_crop(image, height, width):
        image_height, image_width = image.shape[:2]
        if height > image_height or width > image_width:
            raise ValueError(
                f"Crop {(height, width)} is larger than image {(image_height, image_width)}"
            )

        y1 = (image_height - height) // 2
        x1 = (image_width - width) // 2
        return image[y1 : y1 + height, x1 : x1 + width]

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
            "infrared": ImagePairTransformer.save_image(
                images["infrared"], infrared_path
            ),
        }


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def make_transform(enabled, x_bounds, y_bounds):
    if not enabled:
        return Transform()
    return Transform(dx=randint(*x_bounds), dy=randint(*y_bounds))


def transform_record(path, transform):
    return {
        "path": str(path),
        "dx": transform.dx,
        "dy": transform.dy,
        "angle": transform.angle,
    }


def generate_record(args):
    index, pair, split, generation_config, transformer_config = args
    x_bounds = generation_config["bounds"]["x"]
    y_bounds = generation_config["bounds"]["y"]
    visible_generation = generation_config["moving_modality"]["visible"]
    infrared_generation = generation_config["moving_modality"]["infrared"]

    transformer = ImagePairTransformer(
        pair["visible"],
        pair["infrared"],
        **transformer_config,
    )
    if not transformer.is_valid:
        return None

    visible_transform = make_transform(visible_generation, x_bounds, y_bounds)
    infrared_transform = make_transform(infrared_generation, x_bounds, y_bounds)

    transformed = transformer.transform_pair(
        visible=visible_transform,
        infrared=infrared_transform,
    )

    saved_paths = transformer.save_pair(
        transformed,
        REPO_ROOT / f"data_sources/lasher_synthetic/{split}/visible/image_{index}.jpg",
        REPO_ROOT / f"data_sources/lasher_synthetic/{split}/infrared/image_{index}.jpg",
    )

    return {
        "visible": transform_record(saved_paths["visible"], visible_transform),
        "infrared": transform_record(saved_paths["infrared"], infrared_transform),
    }


def generate_split(
    images,
    split,
    generation_config,
    transformer_config,
    max_workers=None,
):
    tasks = [
        (index, pair, split, generation_config, transformer_config)
        for index, pair in enumerate(images)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(
            tqdm(
                executor.map(generate_record, tasks),
                total=len(tasks),
                desc=f"Generating {split}",
            )
        )
    return [record for record in records if record is not None]


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    generation_config = config["model"]["synthetic_generation"]
    transformer_config = {
        "allowed_resolutions": [tuple(item) for item in config["data"]["allowed_resolutions"]],
        "target_height": config["model"]["resolution"]["height"],
        "target_width": config["model"]["resolution"]["width"],
        "canvas_height": generation_config["canvas"]["height"],
        "canvas_width": generation_config["canvas"]["width"],
    }

    train_images = load_jsonl(REPO_ROOT / config["data"]["manifest"]["train"])
    test_images = load_jsonl(REPO_ROOT / config["data"]["manifest"]["test"])

    max_workers = config.get("system", {}).get("workers")

    train_transforms = generate_split(
        train_images,
        "train",
        generation_config,
        transformer_config,
        max_workers=max_workers,
    )
    test_transforms = generate_split(
        test_images,
        "test",
        generation_config,
        transformer_config,
        max_workers=max_workers,
    )

    write_jsonl(REPO_ROOT / "manifests" / "lasher_synthetic_train.jsonl", train_transforms)
    write_jsonl(REPO_ROOT / "manifests" / "lasher_synthetic_test.jsonl", test_transforms)


if __name__ == "__main__":
    main()
