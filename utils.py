import numpy as np
from dataclasses import dataclass, field
import json
from typing import List, Tuple, Union, Optional, Dict
from dataclasses_json import dataclass_json
from supervision import Detections

@dataclass_json
@dataclass
class COCOCategory:
    id: int 
    name: str
    supercategory: str

@dataclass_json
@dataclass
class COCOImage:
    id: int
    dataset_id: int
    category_ids: List[int]
    path: str
    width: int
    height: int
    file_name: str
    annotated: bool
    annotating: List[str]
    num_annotations: int
    metadata: Dict
    deleted: bool
    milliseconds: int
    events: List
    regenerate_thumbnail: bool

@dataclass_json
@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: Tuple[float, float, float, float]
    iscrowd: bool
    isbbox: bool
    color: str
    metadata: Dict

@dataclass_json
@dataclass
class COCOJson:
    images: List[COCOImage] = field(default_factory=list)
    annotations: List[COCOAnnotation] = field(default_factory=list)
    categories: List[COCOCategory] = field(default_factory=list)


class COCOJsonUtility:
    @staticmethod
    def load_coco_json(json_file: str) -> COCOJson:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
        return COCOJson.from_dict(json_data)

    @staticmethod
    def get_annotations_by_image_id(coco_data: COCOJson, image_id: int) -> List[COCOAnnotation]:
        return [annotation for annotation in coco_data.annotations if annotation.image_id == image_id]

    @staticmethod
    def get_annotations_by_image_path(coco_data: COCOJson, image_path: str) -> Optional[List[COCOAnnotation]]:
        image = COCOJsonUtility.get_image_by_path(coco_data, image_path)
        if image:
            return COCOJsonUtility.get_annotations_by_image_id(coco_data, image.id)
        else:
            return None

    @staticmethod
    def get_image_by_path(coco_data: COCOJson, image_path: str) -> Optional[COCOImage]:
        for image in coco_data.images:
            if image.file_name == image_path:
                return image
        return None

    @staticmethod
    def annotations2detections(annotations: List[COCOAnnotation]) -> Dict[str, np.ndarray]:
        class_ids, bboxes = [], []
        for annotation in annotations:
            class_ids.append(annotation.category_id)
            x_min, y_min, width, height = annotation.bbox
            bboxes.append([x_min, y_min, x_min + width, y_min + height])
        
        return {
            "xyxy": np.array(bboxes, dtype=int),
            "class_ids": np.array(class_ids, dtype=int)
        }