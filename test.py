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
bbox_annotator = sv.BoundingBoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
annotate_frame_ground_truth = bbox_annotator.annotate(scene=image_bgr.copy(), detections=ground_truth_detections)

# Predict masks
mask_predictor.set_image(image_rgb)
all_masks = []
for bbox, class_id in zip(ground_truth_detections.xyxy, ground_truth_detections.class_id):
    bbox_array = np.array([bbox], dtype=np.float32)
    masks, scores, logits = mask_predictor.predict(box=bbox_array, multimask_output=True)
    if len(masks) > 0:
        all_masks.extend(masks)
    else:
        print("No masks predicted for bbox:", bbox_array)

# Define a custom color palette
color_palette = sv.ColorPalette([
    sv.Color(255, 0, 0),   # Red
    sv.Color(0, 255, 0),   # Green
    sv.Color(0, 0, 255),   # Blue
    sv.Color(0, 255, 255), # Cyan
    sv.Color(255, 0, 255), # Magenta
    sv.Color(255, 255, 0)  # Yellow
])

# Configure mask annotator with a custom color palette and CLASS lookup
mask_annotator = sv.MaskAnnotator(color=color_palette, color_lookup=sv.ColorLookup.CLASS)

# Annotate and save images
if all_masks:
    masks_array = np.stack(all_masks)
    class_ids_for_masks = [ground_truth['class_ids'][i % len(ground_truth['class_ids'])] for i in range(len(all_masks))]

    detections_from_masks = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks_array),
        mask=masks_array,
        class_id=np.array(class_ids_for_masks)
    )

    annotated_mask_image = mask_annotator.annotate(
        scene=image_bgr.copy(),
        detections=detections_from_masks
    )
else:
    print("No masks generated.")
    annotated_mask_image = image_bgr.copy()

def draw_labels_and_boxes(img, detections, class_names):
    """
    Draw detections and masks on the image
    """
    for det in detections:
        bbox = det.xyxy.astype(int)
        class_id = det.class_id
        # Draw rectangle and label text
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
        label = f"{class_names[class_id]}"
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

annotated_image = draw_labels_and_boxes(image_bgr.copy(), detections_from_masks, class_names=CLASSES)

# cv2.imwrite("bbox_image.jpg", annotate_frame_ground_truth)
# cv2.imwrite("masked_image.jpg", annotated_mask_image)
cv2.imwrite("annotated_image.jpg", annotated_image)

# Display the results
sv.plot_images_grid(
    images=[annotate_frame_ground_truth, annotated_mask_image],
    grid_size=(1, 2),
    titles=['Source Image with BBoxes', 'Segmented Image with Masks']
)
