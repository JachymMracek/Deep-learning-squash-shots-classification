# Deep learning-based classification of squash shots from video sequences

Scripts for creating datasets, training, inference and evaluation of models for squash ball detection, hit detection and shot classification.

## Data preparation

video_to_frames.py – splits videos into individual frames.

get_data_from_videos.py – collects frames from videos for manual annotation (single frames for YOLO, or triples for TrackNet).

get_data_by_yolo.py – automatically creates a YOLO dataset, frames are annotated by an already trained YOLO ball detector.

annotatorMAKING.py – manual annotation tool, bounding boxes are created by clicking on the image and saved in YOLO format.

create_TrackNet_labels.py – creates a dataset in TrackNet format (game/Clip folders + Label.csv) using YOLO ball detection.

create_trackNet_test_video.py – creates short test videos for TrackNet from frame triples.

create_from_triples_test_end_frames.py – selects the last frame of each triple for manual annotation of the test set.

create_convexnet_dataset.py – converts a YOLO dataset into the classifier format (hit / not hit classes).

creator_synthetic_data.py – synthetic data generation of shot trajectories (boast, cross, drive, drop).

get_skeleton_frames_from_full_video.py – saves skeleton frames (player poses + ball) from full videos.

get_real_shot_frames.py – extracts shot trajectory images from videos and sorts them using the ConvNeXt classifier.

frame_sequence_to_path.py – draws a ball trajectory image from a sequence of shot frames.

## Detection and classification

detect_hit_in_video.py – skeleton frame creation and a state machine for hit detection in video.

train_convNeXt.py – fine-tuning of the ConvNeXt Tiny model for shot classification (boast, cross, drive, drop).

convexnet_prediction.py – predicts the shot type and its confidence for a single trajectory image.

infer_path_to_video.py – full inference on a video: draws the ball trajectory and the predicted shot type.

## Evaluation and visualization

test_yolo.py – evaluation of the YOLO ball detector (precision, recall, F1, accuracy).

test_trackNetV1.py – evaluation of TrackNet V1 on test videos.

test_trackNetV5.py – evaluation of TrackNet V5 on test videos.

test_convNeXt.py – evaluation of the ConvNeXt shot classifier (per-class metrics + confusion matrix).

visualize_yolo_bounding_box.py – visualizes the YOLO detector bounding box on a single image.

utils.py – helper functions (video count, train/test split, ball coordinates).

## External Projects Used

TrackNet V1 – unofficial PyTorch implementation: https://github.com/yastrebksv/TrackNet

TrackNet V5 – official implementation: https://github.com/codelancera-offical/TrackNetV5-SDK

External projects were modified for my purposes, such as splitting the dataset correctly, getting ball positions, evaluating models, and training the models. However, they are external projects and need to be cloned separately. I am not the author of these external projects, which is why I do not include their code in the bachelor thesis attachments. For the same reason, I do not include the YouTube images or the models trained on them, as this could raise copyright issues.

## Libraries

If one of the libraries is not installed on your computer, run pip install -r requirements.txt

