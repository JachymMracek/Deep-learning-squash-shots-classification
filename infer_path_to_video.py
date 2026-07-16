import cv2
import detect_hit_in_video
import argparse
from ultralytics import YOLO
import os
import convexnet_prediction
import torch
import get_real_shot_frames


################################################################################
################################################################################
########################### infer shot paths to the video ######################
################################################################################
################################################################################


argument_parser = argparse.ArgumentParser() 
argument_parser.add_argument("--video_path",default = r"", help="Please, write path to the input video")
argument_parser.add_argument("--yolo_hit_model_path",default = r"", help="Please, write path to yolo hit model weights")
argument_parser.add_argument("--yolo_ball_model_path",default =r"", help="Please, write path to yolo ball model weights")
argument_parser.add_argument("--video_output_path",default = r"", help="Please, write path to folder where the output video will be saved")
argument_parser.add_argument("--convexNeXt_path",default = r"", help="Please, write path to the convNeXt model weights")
argument_parser.add_argument("--video_name",default = r"video.mp4", help="Please, what should by the name of the output video. Include .mp4")


def get_ball_position(frame,yolo_model):
    ball_result_box = yolo_model.predict(frame)[0].boxes
      
    if len(ball_result_box.data.tolist()) == 0:
        return None
    
    x_center,y_center,_,_ = ball_result_box.xywhn.tolist()[0]
    
    return (int(x_center*640),int(y_center*360))

def create_video(triple_frames,video_output_folder,video_name,fps ):
    
    height, width, _ = triple_frames[0].shape
    
    video_path = os.path.join(video_output_folder,video_name)
    
    video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
    
    for frame in triple_frames:
        video.write(frame)
    
    video.release()

def draw_path_to_frame(frame,ball_positions,color):
    
    for i in range(len(ball_positions) - 1):
        cv2.line(frame,ball_positions[i],ball_positions[i+1],color,3)

def write_prection_to_frame(frame,ball_positions,convexNeXt_model):
    
    class_number_their_shot_name = {0:"boast",1:"cross",2:"drive",3:"drop"}
    
    height,width, _ = frame.shape
    
    path_shot_frame = get_real_shot_frames.draw_shot_image(ball_positions)
    
    class_predicted, confidence = convexnet_prediction.convexnet_prediction(path_shot_frame,convexNeXt_model)
    
    shot_name = class_number_their_shot_name[class_predicted]
    
    # https://www.geeksforgeeks.org/python/python-opencv-cv2-puttext-method/
    
    cv2.putText(frame,f"{shot_name}: {round(confidence,2)}", (width//2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def draw_path_to_video(video_path,yolo_hit_model,yolo_ball,video_output,convexNeXt_classificator,video_name):
    
    video_capturer = cv2.VideoCapture(video_path)
    
    fps = int(video_capturer.get(cv2.CAP_PROP_FPS)) #infer_on_video.py
    
    shot = detect_hit_in_video.Shot()
    
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    current_color_index = 0
    
    ball_positions = []
    path_frames = []
    
    while True:
            
            readable, frame = video_capturer.read()
            
            if readable:
                frame = cv2.resize(frame,(640,360))
                current_ball_position = get_ball_position(frame,yolo_ball)
                      
                shot.process_frame(frame,current_ball_position,yolo_hit_model)
                print(shot.state)
                
                if shot.state.value == shot.State.END_SHOT.value:
                    current_color_index = (current_color_index + 1) % len(colors)
                    ball_positions = []
                    shot.new_shot()
                    shot.state = shot.State.BEGGINING_SHOT
                
                elif shot.state.value == shot.State.BEGGINING_SHOT.value or shot.state.value == shot.State.DURING_SHOT.value:
                    
                    if current_ball_position is not None:
                            ball_positions.append(current_ball_position)
                    
                    write_prection_to_frame(frame,ball_positions,convexNeXt_classificator)
                    draw_path_to_frame(frame,ball_positions,colors[current_color_index])
                
                path_frames.append(frame)
            
            else:
                break
    
    create_video(path_frames,video_output,video_name,fps)

def main():
    args = argument_parser.parse_args()

    video_path = args.video_path
    yolo_hit_model_path = args.yolo_hit_model_path
    yolo_ball_model_path = args.yolo_ball_model_path
    video_output_path = args.video_output_path
    convexNeXt_path = args.convexNeXt_path
    video_name = args.video_name
    
    yolo_hit_model = YOLO(yolo_hit_model_path)
    yolo_ball_model = YOLO(yolo_ball_model_path)
    convexNeXt_classificator = torch.load(convexNeXt_path,weights_only = False)
    
    draw_path_to_video(video_path,yolo_hit_model,yolo_ball_model,video_output_path,convexNeXt_classificator,video_name)

if __name__ == "__main__":
    main()