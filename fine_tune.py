import os
import random
import numpy as np
import cv2
import torch
from utils import COCOJsonUtility
import supervision as sv
from segment_anything import sam_model_registry, SamPredictor

# Initiate SAM model
CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = 'vit_h'
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

# Dataset directories
DATASET_ROOT = "custom_data/train"
ANNOTATIONS_FILE_NAME = "annotations.json"
IMAGES_DIR_PATH = os.path.join(DATASET_ROOT, "images")
ANNOTATIONS_FILE_PATH = os.path.join(DATASET_ROOT, ANNOTATIONS_FILE_NAME)

# Load COCO data
coco_data = COCOJsonUtility.load_coco_json(json_file=ANNOTATIONS_FILE_PATH)
CLASSES = [category.name for category in coco_data.categories if category.supercategory != 'none']
IMAGES = [image.file_name for image in coco_data.images]

# Randomly select an image
random.seed(1000)
EXAMPLE_IMAGE_NAME = random.choice(IMAGES)
EXAMPLE_IMAGE_PATH = os.path.join(IMAGES_DIR_PATH, EXAMPLE_IMAGE_NAME)

# Load annotations and adjust class IDs for zero-indexing
annotations = COCOJsonUtility.get_annotations_by_image_path(coco_data=coco_data, image_path=EXAMPLE_IMAGE_NAME)
ground_truth = COCOJsonUtility.annotations2detections(annotations=annotations)
ground_truth['class_ids'] = ground_truth['class_ids'] - 1

# Prepare detections object
ground_truth_detections = sv.Detections(
    xyxy=np.array(ground_truth['xyxy'], dtype=int),
    class_id=np.array(ground_truth['class_ids'], dtype=int)
)

# Load image and prepare for inference
image_bgr = cv2.imread(EXAMPLE_IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Annotate using bounding boxes
bbox_annotator = sv.BoundingBoxAnnotator(color=sv.Color.red(), color_lookup=sv.ColorLookup.INDEX)
annotate_frame_ground_truth = bbox_annotator.annotate(scene=image_bgr.copy(), detections=ground_truth_detections)

# Predict masks
mask_predictor.set_image(image_rgb)
all_masks = []
for bbox in ground_truth_detections.xyxy:
    bbox_array = np.array([bbox], dtype=np.float32)  # Ensure bbox is correctly shaped as (1, 4)
    masks, scores, logits = mask_predictor.predict(box=bbox_array, multimask_output=True)
    if len(masks) > 0:  # Check if masks were actually returned
        all_masks.extend(masks)
    else:
        print("No masks predicted for bbox:", bbox_array)


# Annotate and save images
bbox_annotator = sv.BoundingBoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

# Before converting masks to detections, ensure all_masks is a proper NumPy array
if all_masks:
    masks_array = np.stack(all_masks)  # This stacks list of masks into a 3D NumPy array
    detections_from_masks = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks_array),
        mask=masks_array
    )
    annotated_mask_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections_from_masks)
else:
    print("No masks generated.")
    annotated_mask_image = image_bgr.copy()  # Use the original image if no masks were generated


annotated_bbox_image = bbox_annotator.annotate(image_bgr.copy(), ground_truth_detections)
annotated_mask_image = mask_annotator.annotate(image_bgr.copy(), detections_from_masks)
cv2.imwrite("bbox_image.jpg", annotated_bbox_image)
cv2.imwrite("masked_image.jpg", annotated_mask_image)

# Display the results
sv.plot_images_grid(
    images=[annotated_bbox_image, annotated_mask_image],
    grid_size=(1, 2),
    titles=['Source Image with BBoxes', 'Segmented Image with Masks']
)