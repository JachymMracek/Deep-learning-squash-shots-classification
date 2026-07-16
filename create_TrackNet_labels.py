import numpy
import csv
from dataclasses import dataclass
import os
import cv2
from ultralytics import YOLO
import utils

# SOURCES
# https://www.geeksforgeeks.org/python/writing-csv-files-in-python/
# https://www.geeksforgeeks.org/python/image-resizing-using-opencv-python/
# https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/
# https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python


################################################################################
################################################################################
######################    Creates TrackNet V1 frame and csv dataset ############
################################################################################
################################################################################


@dataclass(frozen=True)
class ProgressionImage:
    image:numpy.ndarray
    x_center :float
    y_center:float
    image_index:int

class TrackNetImgsLabels:
    
    def __init__(self) -> None:
        self.yolo_model = YOLO(r"")
    
    @staticmethod
    def _insert_img_label(clip_folder_path,labels_writer,progression_images,VISIBILITY_CONSTANT = "1",STATUS_CONSTANT = "0"):
        
        for progression_image in progression_images:
            file_name = f"{progression_image.image_index}.jpg"
            label_line = [file_name,VISIBILITY_CONSTANT,str(progression_image.x_center),str(progression_image.y_center),STATUS_CONSTANT]
            img_path = os.path.join(clip_folder_path,file_name)

            labels_writer.writerow(label_line)

            cv2.imwrite(img_path,progression_image.image)
    
    def get_best_confidence(self,frame):
       # https://docs.ultralytics.com/modes/predict#key-features-of-predict-mode
       
        detections = self.yolo_model(frame)
        best_box = None
        
        for detection in detections:
            
            if len(detection.boxes) == 0:
                return 0,None
            
            best_idx = int(detection.boxes.conf.argmax().item())
            xyxy = detection.boxes.xyxy.tolist()[best_idx]
            confidence = detection.boxes.conf.max().item()
        
        return confidence,xyxy

    def create_image_progression(self,video_path,clip_folder_path,labels_writer,confidence_threshold = 0.80,
                                GOAL_PROGRESSION_IMAGES_LENGTH = 3,FRAMES_PER_VIDEO=600,CURRENT_FRAME_DISTANCE = 10):

        progression_images = []
        progression_index = 0
        current_frames_distance = 0
        
        video_capturer = cv2.VideoCapture(video_path)
        success = True

        while success:
            success, image = video_capturer.read()

            if progression_index*GOAL_PROGRESSION_IMAGES_LENGTH >= FRAMES_PER_VIDEO:
                return

            if success:
                
                if current_frames_distance < CURRENT_FRAME_DISTANCE:
                    current_frames_distance += 1
                    continue
                
                resized_image = cv2.resize(image,(1280,720))
                confidence, xyxy = self.get_best_confidence(resized_image)
                
                if xyxy is None or confidence_threshold > confidence:
                    progression_images.clear()
                
                
                else:
                    x_left,y_up,x_right,y_down = xyxy
                    
                    x_center = (x_left + x_right) // 2
                    y_center = (y_up + y_down) // 2
                    
                    progression_image_index = GOAL_PROGRESSION_IMAGES_LENGTH*progression_index + len(progression_images)
                    progression_image = ProgressionImage(resized_image,x_center,y_center,progression_image_index)
                    progression_images.append(progression_image)
                
                if len(progression_images) == GOAL_PROGRESSION_IMAGES_LENGTH:
                    TrackNetImgsLabels._insert_img_label(clip_folder_path,labels_writer,progression_images)
                    progression_images.clear()
                    progression_index += 1
                    current_frames_distance = 0
            
            else:
                progression_images.clear()
    
    @staticmethod
    def create_label_csv(clip_folder_path,FRAME_NAME = "file name",VISIBILITY_CLASS = "visibility",
                         X = "x-coordinate",Y="y-coordinate",
                        STATUS = "status",LABELS_FILE_NAME = "Label.csv"):

        labels_file_name = os.path.join(clip_folder_path,LABELS_FILE_NAME)
        labels_file = open(labels_file_name,"w", newline="")
        labels_writer = csv.writer(labels_file,delimiter=",",quotechar="|")
        header_line = [FRAME_NAME,VISIBILITY_CLASS,X,Y,STATUS]
        labels_writer.writerow(header_line)

        return labels_file,labels_writer

    def create_img_labels(self,VIDEO_PATHS = r"C:\Users\jachy\Documents\ShareX\Screenshots\2026-05",GAME_FOLDER_NAME = "game",
                          CLIP_FOLDER_NAME = "Clip1",  # We have just one clip for each game
                          TRACKNET_DATASET = r"C:\Users\jachy\CORRECT_CODES"):
        
        os.makedirs(TRACKNET_DATASET,exist_ok=True)
        
        for i,video_name in enumerate(os.listdir(VIDEO_PATHS)):
            
            number_of_videos = utils.get_number_of_videos(VIDEO_PATHS)
            
            if i >= number_of_videos:
                continue 
            
            video_path = os.path.join(VIDEO_PATHS,video_name)
            game_folder_path = os.path.join(TRACKNET_DATASET,f"{GAME_FOLDER_NAME}{i+1}")
            clip_folder_path = os.path.join(game_folder_path,CLIP_FOLDER_NAME)
            
            os.makedirs(game_folder_path,exist_ok=True)
            os.makedirs(clip_folder_path,exist_ok=True)
            
            labels_file,labels_writer = TrackNetImgsLabels.create_label_csv(clip_folder_path)
            self.create_image_progression(video_path,clip_folder_path,labels_writer)
            labels_file.close()

def main():
    trackNet_imgs_labels = TrackNetImgsLabels()
    trackNet_imgs_labels.create_img_labels()

if __name__ == "__main__":
    main()