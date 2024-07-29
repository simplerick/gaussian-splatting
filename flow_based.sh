#!/bin/bash

# Define the source and target folders
source_folder="$1"
folder="$source_folder/dynamic_flow"
# transform to abs path
source_folder=$(realpath $source_folder)
folder=$(realpath $folder)
# Create the target folders
dynamic_pred_folder="$folder/dynamic_pred"
flow_folder="$folder/flow"

echo "source_folder: $dynamic_pred_folder"
mkdir -p "$dynamic_pred_folder"
mkdir -p "$flow_folder"
# random port
port=$(( ( RANDOM % 1000 )  + 6000 ))


# Predict flow with ProPainter script
(
  cd flow
  conda run -n gaussian_splatting --no-capture-output python run_inference.py "$source_folder/images" "$flow_folder" 2  # 2 is a batch size
  python ../images_to_video.py "$flow_folder/FlowImages_gap1/images" "$flow_folder/flow_gap1.mp4" 5
)

# Navigate to masked-gaussian-splatting directory and execute training, rendering, and video conversion
(
  conda run -n gaussian_splatting --no-capture-output python train_with_transient_and_flow.py -s "$source_folder" -m "$dynamic_pred_folder" \
  --flow "$flow_folder/Flows_gap1/images" --port $port
  conda run -n gaussian_splatting --no-capture-output python render_with_transient.py -m "$dynamic_pred_folder"
  python ../images_to_video.py "$dynamic_pred_folder/train/ours_30000/renders" "$dynamic_pred_folder/video.mp4" 5
  python ../images_to_video.py "$dynamic_pred_folder/train/ours_30000/overlay" "$dynamic_pred_folder/overlay.mp4" 5
  python ../images_to_video.py "$dynamic_pred_folder/train/ours_30000/pred" "$dynamic_pred_folder/pred.mp4" 5
)