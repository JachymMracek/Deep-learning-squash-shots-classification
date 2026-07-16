import os
import cv2
from ultralytics import YOLO
import detect_hit_in_video
import numpy
from torchvision import transforms
import torch
import argparse
from PIL import Image

################################################################################
################################################################################
########################## Get real shots from videos ##########################
################################################################################
################################################################################

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--videos_path",default =r"C:\Users\jachy\Documents\ShareX\Screenshots\2026-05", help="Please, write path to video dataset")
argument_parser.add_argument("--yolo_weights_ball",default =r"D:\FOLDER_OF_FINAL_BACHELOR_THESIS\yolo_ball_dataset\yolo_squash_ball_trained\weights\best.pt", help="Please, write path to yolo ball weights")
argument_parser.add_argument("--convexnet_shot_weigths",default =r"D:\FOLDER_OF_FINAL_BACHELOR_THESIS\ball_hit_model.pth", help="Please, write path to shot convext classificator")
argument_parser.add_argument("--yolo_hit_model",default =r"D:\FOLDER_OF_FINAL_BACHELOR_THESIS\hit_dataset\yolo_hit_ball\weights\best.pt", help="Please, write path to hit yolo model")
argument_parser.add_argument("--output_dataset",default =r"C:\Users\jachy\CORRECT_CODES", help="Please, write path to output dataset")

WIDTH = 640
HEIGHT = 360

def yolo_ball_prediction(frame,yolo_model):
    ball_result_box = yolo_model.predict(frame)[0].boxes
      
    if len(ball_result_box.data.tolist()) == 0:
        return None
    
    x_center,y_center,_,_ = ball_result_box.xywhn.tolist()[0]
    
    return (int(x_center*WIDTH),int(y_center*HEIGHT))

def draw_shot_image(ball_positions):
    shot_image = 255 * numpy.ones((HEIGHT,WIDTH,3), numpy.uint8)
    
    for i in range(len(ball_positions)-1):
        cv2.arrowedLine(shot_image,ball_positions[i],ball_positions[i+1],(255,0,0),4)
    
    if len(ball_positions) >= 2:
        cv2.circle(shot_image,ball_positions[0],8,(0,255,0),-1)
        cv2.circle(shot_image,ball_positions[len(ball_positions)-1],8,(0,0,255),-1)
    
    return shot_image

def save_img(shot_frame,convex_net_shot_classificator,output_dataset,frame_name):
    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # https://www.geeksforgeeks.org/python/convert-opencv-image-to-pil-image-in-python/
    
    test_trasformers = transforms.Compose([transforms.Resize(224),transforms.CenterCrop((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    shot_colored = cv2.cvtColor(shot_frame, cv2.COLOR_BGR2RGB)
    array_colored_shot = Image.fromarray(shot_colored)
    shot_tensor = test_trasformers(array_colored_shot)
    shot_tensor = shot_tensor.unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        outputs = convex_net_shot_classificator(shot_tensor)
        _, preds = torch.max(outputs, 1)

    class_predicted = preds.tolist()[0]
    
    os.makedirs(os.path.join(output_dataset,f"{class_predicted}"),exist_ok=True)
    
    dataset_path = os.path.join(output_dataset,f"{class_predicted}",frame_name)
    cv2.imwrite(dataset_path,shot_frame)

def video(video_dataset_path,yolo_ball_weights,convexnet_shot_weights_path,yolo_hit_weights,output_dataset,WIDTH = 640,HIGHT = 360):
    
    yolo_ball_model = YOLO(yolo_ball_weights)
    yolo_hit_model = YOLO(yolo_hit_weights)
    convexnet_shot_classificator = torch.load(convexnet_shot_weights_path,weights_only=False)
    
    
    for i,video_name in enumerate(os.listdir(video_dataset_path)):
        video_path = os.path.join(video_dataset_path,video_name)
        
        ball_positions = []
        shot = detect_hit_in_video.Shot()
        counter = 0
        frame_num = 0
        
        video_capturer = cv2.VideoCapture(video_path)
        
        while True:
            
            readable, frame = video_capturer.read()
            
            if readable:
                
                frame_num += 1
                frame = cv2.resize(frame,(WIDTH,HIGHT))
                current_ball_position = yolo_ball_prediction(frame,yolo_ball_model)
                shot.process_frame(frame,current_ball_position,yolo_hit_model)
                
                if shot.state.value == shot.State.END_SHOT.value:
                    shot_frame = draw_shot_image(ball_positions)
                    save_img(shot_frame,convexnet_shot_classificator,output_dataset,f"{counter}_{i}.jpg")
                    ball_positions = []
                    shot.state = shot.State.BEGGINING_SHOT
                    shot.new_shot()
                    counter += 1
                
                elif shot.state.value == shot.State.BEGGINING_SHOT.value or shot.state.value == shot.State.DURING_SHOT.value:
                    
                    if current_ball_position is not None:
                        ball_positions.append(current_ball_position)

            if not readable:
                break
        
        video_capturer.release()

def main():
    args = argument_parser.parse_args()
    videos_path = args.videos_path
    yolo_weights_ball = args.yolo_weights_ball
    convexnet_shot_weights_path = args.convexnet_shot_weigths
    yolo_hit_weights = args.yolo_hit_model
    output_dataset = args.output_dataset
    
    video(videos_path,yolo_weights_ball,convexnet_shot_weights_path,yolo_hit_weights,output_dataset)

if __name__ == "__main__":
    main()