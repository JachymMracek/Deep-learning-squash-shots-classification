import os
from ultralytics import YOLO
import argparse
from shapely.geometry import box
from dataclasses import dataclass
import cv2

# https://medium.com/@mervegamzenar/spatial-data-analysis-shapely-fe72d65e63bf

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dataset_path",default = r"",help="Please, write path to dataset with images and labels")
argument_parser.add_argument("--model_weights_path",default = r"", help="Please, write path to yolo weights")
argument_parser.add_argument("--phase",default = "test", help="Please, write phase of the dataset. train/val/test")


@dataclass
class Metrics:
    
    TP: int = 0
    FP: int = 0
    FN: int = 0
    TN: int = 0

def yolo_prediction(frame,yolo_model,NOT_FINDED = ()):
    ball_result_box = yolo_model.predict(frame)[0].boxes
      
    if len(ball_result_box.data.tolist()) == 0:
        return NOT_FINDED
    
    x_center,y_center,width,height = ball_result_box.xywhn.tolist()[0]
    
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    
    return x_min,y_min,x_max,y_max

def change_yolo_metrics(yolo_bounding_box_prediction,label_bounding_box,metrics):
    
    if len(yolo_bounding_box_prediction) != 0 and len(label_bounding_box) == 0:
        metrics.FP += 1
        return
    
    if len(yolo_bounding_box_prediction) == 0 and len(label_bounding_box) == 0:
        metrics.TN += 1
        return
    
    if len(yolo_bounding_box_prediction) == 0 and len(label_bounding_box) != 0:
        metrics.FN += 1
        return
    
    yolo_shapely_box = box(yolo_bounding_box_prediction[0],yolo_bounding_box_prediction[1],yolo_bounding_box_prediction[2],yolo_bounding_box_prediction[3])
    x_center,y_center,width,height = label_bounding_box
    label_bounding_box = box(x_center - width / 2,y_center - height / 2,x_center + width / 2,y_center + height / 2)
    
    if yolo_shapely_box.intersects(label_bounding_box):
        metrics.TP += 1
    
    else:
        metrics.FP += 1
    
    return yolo_shapely_box.intersects(label_bounding_box)
    
def read_box_label(label_file_name):
    
    with open(label_file_name,"r") as label_file:
        bounding_box_coordinates = [float(coordinate) for coordinate in label_file.readline().split(" ")[1:]]
        
    return bounding_box_coordinates

def read_testing_data(dataset_with_hand_test_images,metrics,yolo_weights_path,phase):
    
    yolo = YOLO(yolo_weights_path)
    images_folder = os.path.join(dataset_with_hand_test_images,"images",phase)
    label_folder = os.path.join(dataset_with_hand_test_images,"labels",phase)
    
    for image_name in os.listdir(images_folder):
        
        image_prefix = image_name.split(".")[0]
        print(image_name)
        
        image_path = os.path.join(images_folder,image_name)
        label_path = os.path.join(label_folder,f"{image_prefix}.txt")
        
        image = cv2.imread(image_path)
        
        corners_label_box = read_box_label(label_path)
        yolo_boudning_box_prediction = yolo_prediction(image,yolo)
        
        change_yolo_metrics(yolo_boudning_box_prediction,corners_label_box,metrics)
    
    precision = metrics.TP / (metrics.FP + metrics.TP)
    recall = metrics.TP / (metrics.FN + metrics.TP)
    
    print("precision",precision)
    print("recall",recall)
    print("f1",2*precision*recall / (precision + recall))
    print("accuracy",(metrics.TP + metrics.TN) / (metrics.TP + metrics.TN + metrics.FP + metrics.FN))
        
        
def main():
    
    args = argument_parser.parse_args()
    model_weights_path = args.model_weights_path
    dataset_path = args.dataset_path
    phase = args.phase
    
    metrics = Metrics()
    
    read_testing_data(dataset_path,metrics,model_weights_path,phase)

if __name__ == "__main__":
    main()
