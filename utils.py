import os
    
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