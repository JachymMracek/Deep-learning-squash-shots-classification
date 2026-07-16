import argparse
from ultralytics import YOLO
import cv2


# https://www.geeksforgeeks.org/python/python-draw-rectangular-shape-and-extract-objects-using-opencv/


################################################################################
################################################################################
#################### Visualization of yolo prediction ##########################
################################################################################
################################################################################

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--image_path",default = r"",help="Please, write path to image")
argument_parser.add_argument("--yolo_weights",default = r"",help="Please, write path to yolo weights")

def get_object_coordinates(yolo_model,frame,WIDTH=640,HEIGHT=360):
    
    ball_boxes_prediction = yolo_model.predict(frame)[0].boxes
    
    if len(ball_boxes_prediction.data.tolist()) != 0:
        x_center,y_center,width,height = ball_boxes_prediction.xywhn.tolist()[0]
        
        return (int(x_center*WIDTH),int(y_center*HEIGHT),int(width*WIDTH),int(height*HEIGHT))
    
    return None

def visualize_bounding_box(yolo_model,image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(640,360))
    
    ball_coordinates = get_object_coordinates(yolo_model,image)
    
    if ball_coordinates is not None:
        left_up_corner = (ball_coordinates[0] - ball_coordinates[2]//2,ball_coordinates[1] - ball_coordinates[3]//2)
        right_down_corner = (ball_coordinates[0] + ball_coordinates[2]//2,ball_coordinates[1] + ball_coordinates[3]//2)
        
        cv2.rectangle(image,left_up_corner,right_down_corner,(255,0,0),4,3)
    
    cv2.imshow("",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def main():
    args = argument_parser.parse_args()
    image_path = args.image_path
    yolo_weights = args.yolo_weights

    yolo_model = YOLO(yolo_weights)

    visualize_bounding_box(yolo_model,image_path)


if __name__ == "__main__":
    main()