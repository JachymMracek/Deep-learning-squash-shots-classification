import infer_on_video
import os
import argparse
from dataclasses import dataclass
import cv2


################################################################################
################################################################################
############################ get TrackNetV1 performance ########################
################################################################################
################################################################################



argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dataset_path",default = r"",help="Please, write path to hand anotated last frames")
argument_parser.add_argument("--videos_path",default = r"",help="Please, write path were are videos of size three")


@dataclass
class Metrics:

    TP: int = 0
    FP: int = 0
    FN: int = 0
    TN: int = 0


def test_videos(trackNetV1_label, hand_label, metrics):

    if trackNetV1_label is None and len(hand_label) == 0:
        metrics.TN += 1

    elif trackNetV1_label is None and len(hand_label) != 0:
        metrics.FN += 1

    elif trackNetV1_label is not None and len(hand_label) == 0:
        metrics.FP += 1

    elif trackNetV1_label[0] >= (2*hand_label[0] - hand_label[2]) and trackNetV1_label[0] <= (2*hand_label[2] - hand_label[0]) and trackNetV1_label[1] >= (2*hand_label[1] - hand_label[3]) and trackNetV1_label[1] <= (2*hand_label[3] - hand_label[1]):
        metrics.TP += 1

    else:
        metrics.FP += 1


def get_frame_name(video_name):

    video_index_and_his_index = video_name.split("_")
    last_frame_index = int(video_index_and_his_index[1].split(".")[0])*3
    frame_name = video_index_and_his_index[0] + "_" + str(last_frame_index)

    return frame_name


def get_label_corners(dataset_frame_path, frame_name, video_height, video_width):

    label_frame_path = os.path.join(dataset_frame_path, "labels", "test", f"{frame_name}.txt")

    with open(label_frame_path, "r") as label_file:
        label = [float(bounding_box_label_info) for bounding_box_label_info in label_file.readline().split(" ")[1:]]

    if label == []:
        return ()

    x_center, y_center, width, height = label
    x_left = (x_center - width / 2)*video_width
    x_right = (x_center + width / 2)*video_width
    y_up = (y_center - height / 2)*video_height
    y_down = (y_center + height / 2)*video_height

    return (x_left, y_up, x_right, y_down)


def print_metrics(metrics):

    print(metrics)

    if metrics.TP + metrics.FP == 0 or metrics.TP + metrics.FN == 0:
        print("Cannot compute precision/recall, no positive predictions or no positive labels.")
        return

    precision = metrics.TP / (metrics.FP + metrics.TP)
    recall = metrics.TP / (metrics.FN + metrics.TP)

    print("precision", precision)
    print("recall", recall)

    if precision + recall != 0:
        print("f1", 2*precision*recall / (precision + recall))

    print("accuracy:", (metrics.TP + metrics.TN) / (metrics.TP + metrics.TN + metrics.FP + metrics.FN))


def create(dataset_frame_path, trackNet_test_folder_videos,WIDTH = 640,HEIGHT = 360):

    metrics = Metrics()

    for video_name in os.listdir(trackNet_test_folder_videos):
        video_path = os.path.join(trackNet_test_folder_videos, video_name)

        ball_track = infer_on_video.main("trackNetV1_test_video", video_path) # https://github.com/yastrebksv/TrackNet

        frame_name = get_frame_name(video_name)
        frame_path = os.path.join(dataset_frame_path, "images", "test", f"{frame_name}.jpg")

        frame = cv2.imread(frame_path)
        height, width, _ = frame.shape

        if ball_track[-1] == (None, None):
            last_frame_ball_position = None
            
        else:
            last_frame_ball_position = (int(float(ball_track[-1][0])*width/640),int(float(ball_track[-1][1])*height/360))

        hand_label = get_label_corners(dataset_frame_path, frame_name, height, width)

        test_videos(last_frame_ball_position, hand_label, metrics)

    print_metrics(metrics)


def main():

    args = argument_parser.parse_args()

    dataset_frame_path = args.dataset_path
    dataset_video_path = args.videos_path

    create(dataset_frame_path, dataset_video_path)
if __name__ == "__main__":
    main()