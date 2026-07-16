from enum import Enum
import numpy
import cv2
from ultralytics import YOLO
from dataclasses import dataclass
import math

class HumanPostureBall:
    
    def __init__(self,YOLO_MODEL_NAME = "yolo26x-pose.pt") -> None:
        self.yolo_pose_model = YOLO(YOLO_MODEL_NAME)
        
        self.posture_lines = [(5,6),(5,7),(7,9),(5,11),(11,13),(13,15),(6,8),(6,10),(12,14),(14,16),(11,12)]
        
    def create_image(self,frame,current_ball_position,
                    HEIGHT = 360,WIDTH = 640,BLACK_PIXEL = [0,0,0],WHITE_PIXEL = [255,255,255]):
        
        # https://docs.ultralytics.com/reference/engine/results/
        # https://stackoverflow.com/questions/10465747/how-to-create-a-white-image-in-python
        
        detections = self.yolo_pose_model(frame)
        detection = detections[0]
        all_key_points_people = detection.keypoints.data
        
        image_poses_ball = WHITE_PIXEL[0] * numpy.ones((HEIGHT,WIDTH,3), numpy.uint8)
        
        for key_points_human in all_key_points_people:
            for key_point_index1,key_point_index2 in self.posture_lines:
                
                x_1,y_1,_ = key_points_human[key_point_index1]
                x_2,y_2,_ = key_points_human[key_point_index2]
                
                key_point_1 = (int(x_1),int(y_1))
                key_point_2 = (int(x_2),int(y_2))
                
                cv2.line(image_poses_ball,key_point_1,key_point_2,tuple(BLACK_PIXEL),1)
                
                cv2.circle(image_poses_ball,center = key_point_1,radius = 4,color = tuple(BLACK_PIXEL),
                           thickness = -1)
                
                cv2.circle(image_poses_ball,center = key_point_2,radius = 4,color = tuple(BLACK_PIXEL),
                           thickness = -1)
        
        
        if current_ball_position is not None:
                    
            x_ball_position = current_ball_position[0]
            y_ball_position = current_ball_position[1]
            
            cv2.circle(image_poses_ball,center = (x_ball_position,y_ball_position),radius = 7,
                color = (0,0,255), thickness = -1)
            
        return image_poses_ball

class Shot:
    
    class State(Enum):
        START_VIDEO = 0
        BEGGINING_SHOT = 1
        DURING_SHOT = 2
        END_SHOT = 3
    
    @dataclass
    class EndShotCondition():
        yolo_hit_detection: bool = False
        out_region : bool = False 
    
    def __init__(self) -> None:
        self.state = self.State.START_VIDEO
        self.last_ball_position = None
        self.frames_between_change_condition = 0
    
    def new_shot(self):
        self.frames_between_change_condition = 0
        self.state = self.State.BEGGINING_SHOT
        
    def is_ball_hit(self,frame,yolo_hit_ball_model):
        
        hit_ball_reult_box = yolo_hit_ball_model.predict(frame)[0].boxes
        
        if len(hit_ball_reult_box.data.tolist()) != 0:
            return True
    
        return False
    
    def is_ball_out_region(self,current_ball_position,THRESHOLD = 0):
        
        if current_ball_position is None or self.last_ball_position is None:
            return False
        
        distance = math.sqrt((self.last_ball_position[0] - current_ball_position[0])**2 + (self.last_ball_position[1] - current_ball_position[1])**2)
        
        return distance > THRESHOLD
    
    def process_frame(self,frame,current_ball_position,yolo_hit_ball_model,FRAMES_BETWEEN_CONDITIONS = 2):
        
        skeleton_ball_frame = HumanPostureBall().create_image(frame,current_ball_position)
        
        is_ball_hit = self.is_ball_hit(skeleton_ball_frame,yolo_hit_ball_model)
        
        if self.state.value == self.State.START_VIDEO.value and is_ball_hit:
            self.state = self.State.BEGGINING_SHOT
        
        elif self.state == self.State.BEGGINING_SHOT and is_ball_hit and self.frames_between_change_condition < FRAMES_BETWEEN_CONDITIONS:
            self.frames_between_change_condition = 0
        
        elif self.state == self.State.BEGGINING_SHOT and is_ball_hit and self.frames_between_change_condition >= FRAMES_BETWEEN_CONDITIONS:
            self.state = self.State.DURING_SHOT
            self.frames_between_change_condition = 0
            
        elif self.state == self.State.BEGGINING_SHOT and not is_ball_hit:
            self.frames_between_change_condition += 1
        
        elif self.state == self.State.DURING_SHOT and is_ball_hit and self.frames_between_change_condition >= FRAMES_BETWEEN_CONDITIONS:
            self.state = self.State.END_SHOT
        
        elif self.state == self.State.DURING_SHOT and is_ball_hit and self.frames_between_change_condition < FRAMES_BETWEEN_CONDITIONS:
            self.frames_between_change_condition = 0
        
        elif self.state == self.State.DURING_SHOT and not is_ball_hit:
            self.frames_between_change_condition += 1
