from get_data_from_videos import VideoReader
import os, cv2
from overrides import override
import argparse
from ultralytics import YOLO
import yaml
import utils


################################################################################
################################################################################
############################## Get frames by Yolo ##############################
################################################################################
################################################################################

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--path_videos_input",default = r"", help="Please, write path to video dataset")
argument_parser.add_argument("--path_frames_output",default = r"", help="Please, write path to folder where images will be saved.")
argument_parser.add_argument("--yolo_path_weights",default = r"", help="Please, write a path to yolo weights")

class AutomativeFrameCollector(VideoReader):
    
    """Automatically generate dataset for next yolo training"""
    
    IMAGE_FOLDER_NAME = "images"
    LABELS_FOLDER_NAME = "labels"
    PHRASES = ["train","val"]
    
    def __init__(self,video_path,video_id,count_of_videos,path_dataset_output,yolo_model) -> None:
        super().__init__(video_path, video_id)
        
        self.numbers_of_video = count_of_videos
        self.yolo_model = yolo_model
        self.create_dataset_folder(path_dataset_output)
        self.yolo_result_box = None
        self.empty_frames = 0
    
    """
    creates dataset and nested folders where images and labels will be stored

    Arguments:
        path_dataset_output (string): output path where the dataset will be created
    
    """    
    
    def create_dataset_folder(self,path_dataset_output):
        
        os.makedirs(path_dataset_output,exist_ok=True)
        
        images_path_folder = os.path.join(path_dataset_output,self.IMAGE_FOLDER_NAME)
        labels_path_folder = os.path.join(path_dataset_output,self.LABELS_FOLDER_NAME)
        
        os.makedirs(images_path_folder,exist_ok=True)
        os.makedirs(labels_path_folder,exist_ok=True)
        
        for phase in self.PHRASES:
            nested_image_path = os.path.join(images_path_folder,phase)
            nested_label_path = os.path.join(labels_path_folder,phase)
            
            os.makedirs(nested_image_path,exist_ok=True)
            os.makedirs(nested_label_path,exist_ok=True)
    
    """
    decides if frame satisfied following rules 

    Arguments:
        frame (numpy.ndarray): potencial frame in the dataset
    
    Returns:
        boolean which is a result of frame satisfaction
    """       
    
    @override
    def is_frame_acceptable(self,frame,MUST_DISTANCE_BETWEEN_FRAMES = 60,CORRECT_ACCEPT_THRESHOLD = 0.8,EMPTY_FRAMES_THRESHOLD = 1):
        
        self.yolo_result_box = self.yolo_model.predict(frame)[0].boxes
        
        is_accepted = self.current_distance > MUST_DISTANCE_BETWEEN_FRAMES and ((len(self.yolo_result_box.data.tolist()) != 0 and self.yolo_result_box.conf.tolist()[0] >= CORRECT_ACCEPT_THRESHOLD) 
                        or (len(self.yolo_result_box.data.tolist()) == 0 and self.empty_frames < EMPTY_FRAMES_THRESHOLD))

        if len(self.yolo_result_box.data.tolist()) == 0  and self.empty_frames < EMPTY_FRAMES_THRESHOLD and self.current_distance > MUST_DISTANCE_BETWEEN_FRAMES:
            self.empty_frames += 1
        
        return is_accepted

    """
    saves frame if it satisfy following rules

    Arguments:
        frame (numpy.ndarray): potencial frame in the dataset
        output_path_dataset (string): dataset where frames are stored
    """   

    def save_img(self,frame,output_path_dataset,TRAIN_QUOTIENT = 0.8,VAL_QUOTIENT = 0.1,DEFAULT_CLASS = 0):
        
        img_name = self.get_image_name()
        img_prefix = img_name.split(".")[0]
        
        if self.video_id < int(self.numbers_of_video*TRAIN_QUOTIENT):
            img_path = os.path.join(output_path_dataset,self.IMAGE_FOLDER_NAME,self.PHRASES[0],img_name)
            label_file_name = os.path.join(output_path_dataset,self.LABELS_FOLDER_NAME,self.PHRASES[0],f"{img_prefix}.txt")
            print(img_path)
        
        elif self.video_id < int(self.numbers_of_video*(TRAIN_QUOTIENT+VAL_QUOTIENT)):
            img_path = os.path.join(output_path_dataset,self.IMAGE_FOLDER_NAME,self.PHRASES[1],img_name)
            label_file_name = os.path.join(output_path_dataset,self.LABELS_FOLDER_NAME,self.PHRASES[1],f"{img_prefix}.txt")
        
        cv2.imwrite(img_path,frame)
        
        if len(self.yolo_result_box.data.tolist()) != 0:
            best_boudning_box_index = self.yolo_result_box.conf.argmax()
            x_center,y_center,width,height = self.yolo_result_box.xywhn.tolist()[best_boudning_box_index]
            label = f"{DEFAULT_CLASS} {x_center} {y_center} {width} {height}"
        
        else:
            label = ""
        
        with open(label_file_name, "w") as label_file:
            label_file.write(label)


"""
Creates yolo yaml file

Arguments:
    dataset_path (string): path of dataset where yaml will be save

"""   

def create_data_yaml(dataset_path,DATASET_NAME = "data.yaml"):
    
    data_content = {"path":dataset_path, "train":"images/train","val":"images/val","names":{0:"squashBall"}}
    
    data_path = os.path.join(dataset_path,DATASET_NAME)
    
    with open(data_path,"w") as data_yaml:
        yaml.dump(data_content,data_yaml)

"""
Creates dataset in yolo format for next training

Arguments:
    path_to_videos_string (string) : path where squash videos are located
    path_frames_output (string) :  path where images will be located
    yolo_model_weights_path (string): path to yolo which detect squash balls in the frames

"""        

def create_dataset_for_yolo_automative_annotations(path_to_videos_string,path_frames_output,yolo_model_weights_path,TEST_QUOTIENT_BEFORE = 0.9):
    
    yolo_model = YOLO(yolo_model_weights_path)
    
    count_videos = utils.get_number_of_videos(path_to_videos_string)
    
    for video_id,video_name in enumerate(os.listdir(path_to_videos_string)):
        
        if video_id >= int(count_videos*TEST_QUOTIENT_BEFORE):
            break
        
        video_path = os.path.join(path_to_videos_string,video_name)
        
        automative_frame_collector = AutomativeFrameCollector(video_path,video_id,count_videos,path_frames_output,yolo_model)
        
        for frame in automative_frame_collector.get_new_frame():
            automative_frame_collector.save_img(frame,path_frames_output)
    
    create_data_yaml(path_frames_output)

def main():
    
    args = argument_parser.parse_args()

    path_videos_input = args.path_videos_input
    path_frames_output = args.path_frames_output
    yolo_model_weights_path = args.yolo_path_weights
    
    create_dataset_for_yolo_automative_annotations(path_videos_input,path_frames_output,yolo_model_weights_path)
    
if __name__ == "__main__":
    main()
