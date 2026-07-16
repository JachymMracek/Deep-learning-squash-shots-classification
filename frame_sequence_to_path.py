import os
import  cv2
import argparse
import numpy
from ultralytics import YOLO
import utils

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--root_test_shots_folder",default = r"", help="Please, write path where are frames")
argument_parser.add_argument("--output_folder",default = r"", help="Please, add path where the shot will be saved")
argument_parser.add_argument("--yolo_ball_weights",default = r"", help="Please, write path to yolo ball weights")

def create_shot_image(ball_positions,image_path):
    image_boast_trajectory_image = 255 * numpy.ones((360,640,3), numpy.uint8)
    
    for i in range(len(ball_positions)-1):  
        cv2.arrowedLine(image_boast_trajectory_image,ball_positions[i],ball_positions[i+1],(255,0,0),4)
    
    cv2.circle(image_boast_trajectory_image,ball_positions[0],8,(0,255,0),-1)
    cv2.circle(image_boast_trajectory_image,ball_positions[len(ball_positions)-1],8,(0,0,255),-1)
    
    cv2.imwrite(image_path,image_boast_trajectory_image)
    

def create_testing_videos(root_test_shots_folder,output_folder,yolo_ball_dataset):
    
    yolo_model = YOLO(yolo_ball_dataset)
    
    for shot_type_folder_name in os.listdir(root_test_shots_folder):
        shot_type_frames_path = os.path.join(root_test_shots_folder,shot_type_folder_name)
        shot_type_draw_path = os.path.join(output_folder,shot_type_folder_name)
        
        
        os.makedirs(shot_type_draw_path,exist_ok=True)
        shot_index = 0
        
        for folder_shot_frames in os.listdir(shot_type_frames_path):
            folder_frames_video = os.path.join(shot_type_frames_path,folder_shot_frames)
            
            ball_positions = []

            for frame_name in os.listdir(folder_frames_video):
                frame_path = os.path.join(folder_frames_video,frame_name)
                frame = cv2.imread(frame_path)
                ball_coordinates = utils.get_ball_coordinates(yolo_model,frame)
                print(ball_coordinates)
                
                if ball_coordinates is not None:
                    ball_positions.append(ball_coordinates)
            
            draw_image_path = os.path.join(shot_type_draw_path,f"{shot_index}.jpg")
            
            if len(ball_positions) != 0:    
                create_shot_image(ball_positions,draw_image_path)
            
            shot_index += 1

def main():
    args = argument_parser.parse_args()
    
    root_test_shots_folder = args.root_test_shots_folder
    output_folder = args.output_folder
    yolo_ball_weights = args.yolo_ball_weights
    
    os.makedirs(output_folder,exist_ok=True)
    
    create_testing_videos(root_test_shots_folder,output_folder,yolo_ball_weights)

if __name__ == "__main__":
    main()
