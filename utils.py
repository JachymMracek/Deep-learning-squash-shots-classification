import cv2
from dataclasses import dataclass
import os

################################################################################
################################################################################
############################## help methods ####################################
################################################################################
################################################################################


@dataclass(frozen=True)  
class Click:
    x: int
    y: int

def middle(point_value1,point_value2 = None):
    
    if point_value2 is None:
        return point_value1 / 2
    
    return (point_value1 + point_value2) / 2

def click_event(event, x, y, flags, params:list):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        click = Click(x,y)
        params.append(click)

def click_on_image(img,click_count,clicks):
    # https://www.geeksforgeeks.org/python/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    
    cv2.imshow("image",img) 
    cv2.setMouseCallback('image',click_event,clicks)

    for _ in range(click_count):
        cv2.waitKey(0)
    
      
    cv2.destroyAllWindows()
    
def get_number_of_videos(video_folder_path):
    
    count_videos = 0
    
    for _ in os.listdir(video_folder_path):
        count_videos += 1
    
    return count_videos

def is_video_for_phase(video_idx,count_videos,phase,TEST_PHASE = "test",NOT_TEST_PHASE = "train",NOT_TEST_QUOTIENT = 0.9):
    
    if video_idx < count_videos*NOT_TEST_QUOTIENT and phase == NOT_TEST_PHASE:
        return True
    
    elif video_idx >= count_videos*NOT_TEST_QUOTIENT and phase == TEST_PHASE:
        return True
    
    return False

def get_ball_coordinates(yolo_model,frame):
    ball_boxes_prediction = yolo_model.predict(frame)[0].boxes
    
    if len(ball_boxes_prediction.data.tolist()) != 0:
        x_center,y_center,_,_ = ball_boxes_prediction.xywhn.tolist()[0]
        
        return (int(x_center*640),int(y_center*360))
    
    return None