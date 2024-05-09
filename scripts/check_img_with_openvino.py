from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image_as_pil
from sahi.predict import get_sliced_prediction
import time
import os

threshold = 0.0

def check_image(img_path, model_path_dir, device_to_run, slice_width, slice_height):
    # Start the timer
    start_time = time.time()

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='openvino',
        model_path=model_path_dir,
        confidence_threshold=0.0,
        device=device_to_run,  # or 'cuda:0'
    )

    image = read_image_as_pil(img_path)

    results = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    image_name = os.path.splitext(os.path.basename(img_path))[0]

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    # uncomment this to save predicted image
    # results.export_visuals(export_dir=export_dir_path, file_name=image_name)


img_path = './builds/test_images/1.png'
model_path_dir = './builds/full/last_6_openvino_model/'
device_to_run = 'cpu' # or 'cuda:0'

check_image(img_path, model_path_dir, device_to_run, 250, 250)