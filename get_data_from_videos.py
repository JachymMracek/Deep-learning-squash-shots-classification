import argparse
from pathlib import Path
import cv2
import os
from abc import ABC, abstractmethod
from overrides import override
from dataclasses import dataclass
import utils

# SOURCES:
# https://www.geeksforgeeks.org/python/command-line-arguments-in-python/
# https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/
# https://pytutorial.com/python-pathlib-iterdir-explained/
# https://dev.to/lifeportal20002010/python-block-comments-best-practices-shortcuts-and-docstrings-explained-342f
# https://www.geeksforgeeks.org/python/abstract-classes-in-python/
# https://pypi.org/project/overrides/


################################################################################
################################################################################
######### This file gets data for training and testing tracknet and yolp #######
################################################################################
################################################################################

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--path_videos_input",default = r"", help="Please, write path to videos dataset")
argument_parser.add_argument("--path_frames_output",default = r"", help="Please, write path to folder where images will be saved.")
argument_parser.add_argument("--model_type",default = r"yolo", help="Choose between tracknet triple or yolo single images collector. Modes are trackNet/yolo")
argument_parser.add_argument("--phase",default = r"train", help="User can choose two options test/train. This parameter decide from which videos are data collect it")

class VideoReader(ABC):
    
    """
    Class which iterate over given videos and processes video frames
    """
    
    GOAL_NUMBER_OF_FRAMES_VIDEO = 63
    
    def __init__(self,video_path,video_id) -> None:
        self.video_id = video_id # constant
        self.taken_frames_from_video = 0
        self.current_distance = 0
        self.video_path = video_path
        self.video_capturer = cv2.VideoCapture(video_path)
    
    # frame was accepted, distance between current image and acepted image is zero.
    def _update_frame_was_accepted(self):
        self.current_distance = 0
        self.taken_frames_from_video += 1
    
    # distance between frames get one image larger
    def _update_frame_was_not_accepted(self):
        self.current_distance += 1
    
    # returns name of image. In image folder image will be named like this method return
    def get_image_name(self):
        return f"{self.video_id}_{self.taken_frames_from_video}.jpg"
    
    """
        Decide if frame is acceptable.
        
        Returns:
            bool satisfying frame 
    """
    
    @abstractmethod
    def is_frame_acceptable(self,frame,MUST_DISTANCE_BETWEEN_FRAMES = 200):
        pass
    
    """
        iterate over frames in video and if the frame is accept it then method returns given frame.
        
        Returns:
            numpy.ndarray satisfying frame 
    """

    def get_new_frame(self):     
        
        while True: 
                   
            new_frame_detected, frame = self.video_capturer.read()
                    
            if not new_frame_detected or self.taken_frames_from_video == self.GOAL_NUMBER_OF_FRAMES_VIDEO:
                break
                    
            elif self.is_frame_acceptable(frame):
                self._update_frame_was_accepted()
                yield frame
                    
            else:
                self._update_frame_was_not_accepted()

class HandFrameCollector(VideoReader):
    
    """
    Collect data which will be then annotate it by hand
    """
    
    # For single data frame data collecting
    @dataclass(frozen = True)
    class YoloMode:
        pass
    
    # For triples data frame data collecting
    @dataclass
    class TrackNetMode:
        current_taken_frames:int = 0
    
    def __init__(self, video_path, video_id,mode,YOLO_MODE = "yolo",TRACKNET_MODE = "trackNet") -> None:
        super().__init__(video_path, video_id)
        
        self.mode = None
        
        if mode == YOLO_MODE:
            self.mode = self.YoloMode()
        
        elif mode == TRACKNET_MODE:
            self.mode = self.TrackNetMode()
    
    # Frame is accepted when distance between previous one is responsible so the model does not contain in dataset same frames
    @override
    def is_frame_acceptable(self,frame,MUST_DISTANCE_BETWEEN_FRAMES = 200,TRACKNET_NUMBER_FRAMES = 3):
        
        if isinstance(self.mode,self.TrackNetMode):
            
            if self.mode.current_taken_frames == TRACKNET_NUMBER_FRAMES:
                self.mode.current_taken_frames = 0
                return False
            
            elif (self.mode.current_taken_frames == 0 and self.current_distance > MUST_DISTANCE_BETWEEN_FRAMES) or self.mode.current_taken_frames != 0:
                self.mode.current_taken_frames += 1
                return True
            
            elif self.mode.current_taken_frames == 0 and self.current_distance < MUST_DISTANCE_BETWEEN_FRAMES:
                return False
        
        elif isinstance(self.mode,self.YoloMode):
            return self.current_distance > MUST_DISTANCE_BETWEEN_FRAMES

"""
Creates dataset which will consists images for yolo hand annotations

Arguments:
    video_folder_path (string) : path where squash videos are located
    path_frames_output (string) :  path where images will be located
    mode (string): choose which type of data are taken. Single frames or triples
    phase (string): describes from which videos frames must be collect it. Depends test or not test phase

"""

def create_dataset_for_yolo_hand_annotations(video_folder_path,path_frames_output,mode,phase):
    
    count_videos = utils.get_number_of_videos(video_folder_path)
    
    for video_id,video_name in enumerate(os.listdir(video_folder_path)):
        
        video_path = os.path.join(video_folder_path,video_name)
        
        if not utils.is_video_for_phase(video_id,count_videos,phase):
            continue
        
        video_reader = HandFrameCollector(video_path,video_id,mode)
        
        for frame in video_reader.get_new_frame():
                
            img_name = video_reader.get_image_name()
            frame_path = os.path.join(path_frames_output,img_name)
            cv2.imwrite(frame_path,frame)
                
        if video_reader.taken_frames_from_video >= VideoReader.GOAL_NUMBER_OF_FRAMES_VIDEO:
            continue

def main():
    
    args = argument_parser.parse_args()
    
    mode = args.model_type
    phase = args.phase
    
    os.makedirs(args.path_frames_output,exist_ok=True)
    
    create_dataset_for_yolo_hand_annotations(args.path_videos_input,args.path_frames_output,mode,phase)

if __name__ == "__main__":
    main()
