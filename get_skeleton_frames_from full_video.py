import argparse
import os
import cv2
from detect_hit_in_video import HumanPostureBall
from ultralytics import YOLO
import utils



################################################################################
################################################################################
########### Script which save frames of players pose and squash ball ###########
################################################################################
################################################################################


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--videos_path",default = r"",help="Please, write path to video dataset")
argument_parser.add_argument("--yolo_weights_path",default = r"",help="Please, write path of yolo squash ball detector")
argument_parser.add_argument("--output_path",default = r"",help="Please, write path where frames will be saved")
argument_parser.add_argument("--phase",default =r"",help="Please, typed you phase. train or test")


"""
    Iterate frames over video
        
    Arguments:
        video_path (string): Video which is iterated by this method
        yolo_model (YOLO): squash ball detector
        output_path (string): string path where frame with pose with ball are saved
"""

def frames_video(video_path,yolo_model,output_path,video_index):
    
    video_capturer = cv2.VideoCapture(video_path)

    frame_index = 0
    
    while True:
        
        readable, frame = video_capturer.read()
        
        if readable:
            current_ball_position = utils.get_ball_coordinates(yolo_model,frame)
            resized_frame = cv2.resize(frame,(640,360))
            skeleton_ball_frame = HumanPostureBall().create_image(resized_frame,current_ball_position)
            frame_path = os.path.join(output_path,f"{video_index}_{frame_index}.jpg")
            cv2.imwrite(frame_path,skeleton_ball_frame)
            frame_index += 1
        
        else:
            break

    video_capturer.release()

"""
    Iterate over videos in video folder
        
    Arguments:
        videos_path (string): path to folder where are squash videos
        yolo_model (YOLO): Squash ball detector
        phase (string): Users choice parameter if user wants to collect data from test or train videos
        output_path (string): Where final frames are stored
"""

def iterate_videos(videos_path,yolo_model,phase,output_path):
    
    count_of_videos = utils.get_number_of_videos(videos_path)
    
    for video_index,video_name in enumerate(os.listdir(videos_path)):
        video_path = os.path.join(videos_path,video_name)
        
        if utils.is_video_for_phase(video_index,count_of_videos,phase):
            frames_video(video_path,yolo_model,output_path,video_index)

def main():
    
    args = argument_parser.parse_args()
    videos_path = args.videos_path
    yolo_weights_path = args.yolo_weights_path
    output_path = args.output_path
    phase = args.phase
    
    yolo_model = YOLO(yolo_weights_path)
    
    os.makedirs(output_path,exist_ok=True)
    
    iterate_videos(videos_path,yolo_model,phase,output_path)

if __name__ == "__main__":
    main()
