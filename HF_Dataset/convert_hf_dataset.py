from datasets import Dataset, load_dataset, Features, ClassLabel, Array2D, Array3D, Sequence, Value
import json
import os
import numpy as np
from PIL import Image

categories = ['acne', 'darkcircles', 'openpores', 'finelines', 'wrinkle', 'pigmentation']

features = Features({
    'image': Array3D(dtype="uint8", shape=(512, 512, 3)),
    'category_ids': Sequence(ClassLabel(names=categories)),
    'segmentation': Sequence(Sequence(Sequence(Value(dtype='float32')))),
    'bbox': Sequence(Array2D(dtype='float32', shape=(4,))),
    'area': Sequence(Value('float32')),
    'iscrowd': Sequence(Value('bool'))
})

def load_image(image_path):
    with Image.open(image_path) as img:
        return np.array(img)

def parse_annotations(data_dir, split):
    annotations_path = os.path.join(data_dir, split, 'annotations.json')
    with open(annotations_path, 'r') as file:
        data = json.load(file)
        images = data['images']
        annotations = data['annotations']

    # Initialize lists for each attribute
    images_data = []
    category_ids = []
    segmentations = []
    bboxes = []
    areas = []
    iscrowds = []

    for image_info in images:
        image_path = os.path.join(data_dir, split, 'images', image_info['file_name'])
        if not os.path.exists(image_path):
            continue

        # Filter annotations for this image
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_info['id']]
        if not image_annotations:
            continue

        # Append data for each image
        images_data.append(load_image(image_path))
        category_ids.append([ann['category_id'] - 1 for ann in image_annotations])
        segmentations.append([ann['segmentation'] for ann in image_annotations])
        bboxes.append([ann['bbox'] for ann in image_annotations])
        areas.append([ann['area'] for ann in image_annotations])
        iscrowds.append([ann['iscrowd'] for ann in image_annotations])

    # Create a dictionary that matches the features structure
    data_dict = {
        'image': images_data,
        'category_ids': category_ids,
        'segmentation': segmentations,
        'bbox': bboxes,
        'area': areas,
        'iscrowd': iscrowds
    }

    return data_dict

if __name__ == '__main__':
    data_dir = '../custom_data/'
    train_data = parse_annotations(data_dir, 'train')
    val_data = parse_annotations(data_dir, 'val')

    train_dataset = Dataset.from_dict(train_data, features=features)
    val_dataset = Dataset.from_dict(val_data, features=features)

    # Example usage
    print(train_dataset[0])