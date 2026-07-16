import os
import shutil
import argparse

# https://www.geeksforgeeks.org/python/python-shutil-copy-method/


################################################################################
################################################################################
############### Dataset converter from YOLO format to ConvexNeXt ###############
################################################################################
################################################################################


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--yolo_dataset_path",default = r"",help="Please, write path to root yolo dataset")
argument_parser.add_argument("--convexnext_dataset_path",default = r"",help="Please, write output path where ConvexNeXt dataset will be created")


"""
    Returns source frame path in YOLO dataset and target frame path in ConvNeXt dataset
        
    Arguments:
        file_tag (string): tag is name of the image before postfix.
        yolo_dataset_path (string): path of root YOLO dataset path.
"""

def get_frame_path_and_taret_path(file_tag,yolo_dataset_path,convexnet_dataset_path,phase,class_name,IMAGES_FOLDER_NAME = "images"):
    
    file_name = f"{file_tag}.jpg"
    current_frame_path = os.path.join(yolo_dataset_path,IMAGES_FOLDER_NAME,phase,file_name)
    target_frame_path = os.path.join(convexnet_dataset_path,phase,class_name,file_name)
    
    return (current_frame_path,target_frame_path)


"""
    Converts YOLO dataset to ConvNeXt dataset
        
    Arguments:
        yolo_dataset_path (string): path to the YOLO dataset
        convexnet_dataset_path (string): output path where converted dataset will be.
"""

def convert_yolo_dataset_to_convexnet(yolo_dataset_path,convexnet_dataset_path,LABELS_FOLDER_NAME = "labels",PHASES = ["train","val"],HIT_CLASS = "hit",NOT_HIT_CLASS = "not hit"):
    
    labels_folder = os.path.join(yolo_dataset_path,LABELS_FOLDER_NAME)
    
    os.makedirs(convexnet_dataset_path,exist_ok=True)
    
    for phase in PHASES:
        convexnet_phase_path = os.path.join(convexnet_dataset_path,phase)
        convexnet_phase_hit_path = os.path.join(convexnet_phase_path,HIT_CLASS)
        convexnet_phase_not_hit_path = os.path.join(convexnet_phase_path,NOT_HIT_CLASS)
        
        phase_labels_path = os.path.join(labels_folder,phase)
        
        os.makedirs(convexnet_phase_path,exist_ok=True)
        os.makedirs(convexnet_phase_hit_path,exist_ok=True)
        os.makedirs(convexnet_phase_not_hit_path,exist_ok=True)
         
        for label_file_name in os.listdir(phase_labels_path):
        
            label_file_path = os.path.join(phase_labels_path,label_file_name)
        
            file_tag = label_file_name.split(".")[0]
        
            with open(label_file_path,"r") as open_label_file:
                yolo_label = open_label_file.readline()

            if yolo_label == "":
                current_frame_path,target_frame_path = get_frame_path_and_taret_path(file_tag,yolo_dataset_path,convexnet_dataset_path,phase,NOT_HIT_CLASS)
            
            else:
                current_frame_path,target_frame_path = get_frame_path_and_taret_path(file_tag,yolo_dataset_path,convexnet_dataset_path,phase,HIT_CLASS)
            
            print(current_frame_path,target_frame_path)
            shutil.copy(current_frame_path,target_frame_path)

def main():
    args = argument_parser.parse_args()

    yolo_dataset_path = args.yolo_dataset_path
    convexnext_dataset_path = args.convexnet_dataset_path
    
    convert_yolo_dataset_to_convexnet(yolo_dataset_path,convexnext_dataset_path)
    
if __name__ == "__main__":
    main()
