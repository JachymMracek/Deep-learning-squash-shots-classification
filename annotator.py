import cv2
from dataclasses import dataclass
import argparse
import os
import shutil
import yaml
import help_methods

# https://www.geeksforgeeks.org/python/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
# https://www.geeksforgeeks.org/python/how-to-iterate-over-files-in-directory-using-python/
# https://docs.ultralytics.com/datasets/detect/
# https://www.geeksforgeeks.org/python/data-classes-in-python-an-introduction/
# https://www.geeksforgeeks.org/python/define-custom-exceptions-in-python/
# https://www.geeksforgeeks.org/python/python-opencv-cv2-rectangle-method/
# https://www.geeksforgeeks.org/python/reading-and-writing-yaml-file-in-python/
# https://medium.com/@basit.javed.awan/resizing-multiple-images-and-saving-them-using-opencv-518f385c28d3

parser = argparse.ArgumentParser()
parser.add_argument("--yolo_dataset",default=r"",help="Please, write output path where yolo dataset will be created")
parser.add_argument("--input_folder",default=r"",help="Please, write path to folder with frames to annotate")
parser.add_argument("--mode",default="test",help = "choose between train and test")

@dataclass(frozen=True)
# https://www.geeksforgeeks.org/python/data-classes-in-python-an-introduction/
class Click:
    x: int
    y: int

class ImageData:

    class FolderContainsWrongFile(Exception):

        def __init__(self,wrong_item,MESSAGE="Folder contains wrong file or folder:"):
            super().__init__(MESSAGE)

            self.message = MESSAGE
            self.wrong_item = wrong_item
        
        def __str__(self):
            return f"{self.message} {self.wrong_item}"
        
    
    class WrongInputFolder(Exception):

        def __init__(self,name_input,MESSAGE="Input is not folder:"):
            super().__init__(MESSAGE)

            self.message = MESSAGE
            self.name_input = name_input
        
        def __str__(self):
            return f"{self.message} {self.name_input}"
    
    class WrongFileName(Exception):

        def __init__(self,fileName,MESSAGE="Prefix of file is not digital:"):
            super().__init__(MESSAGE)

            self.message = MESSAGE
            self.fileName = fileName
        
        def __str__(self):
            return f"{self.message} {self.fileName}"
    
    @dataclass(frozen=True)
    class Data:
        yolo_format: str
        img_path: str

    @classmethod
    def _middle(cls,point_value1,point_value2):
        return (point_value1 + point_value2) / 2
    
    @classmethod
    def _difference(cls,point_value1,point_value2):
        return abs(point_value1 - point_value2)
    
    @classmethod
    def _normalize(cls,value,divider):
        return value / divider

    @classmethod
    def _get_yolo_format(cls,class_number,click1:Click,click2:Click,height_image,width_image):
        x_center = cls._normalize(cls._middle(click1.x,click2.x),width_image)
        y_center = cls._normalize(cls._middle(click1.y,click2.y),height_image)
        width = cls._normalize(cls._difference(click1.x,click2.x),width_image)
        height = cls._normalize(cls._difference(click1.y,click2.y),height_image)

        return f"{class_number} {x_center} {y_center} {width} {height}"
    
    @classmethod
    def _visualization_annotation(cls,image_path,click_1:Click,click_2:Click):
        image = cv2.imread(image_path)
        cv2.rectangle(image,(click_1.x,click_1.y),(click_2.x,click_2.y), (255, 0, 0), 2)
        
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def _anotate_file(cls,class_number,image_path,CLICK_COUNT=2):
        clicks = []

        clickable_image = cv2.imread(image_path,1)
        help_methods.click_on_image(clickable_image,CLICK_COUNT,clicks)

        height_image,width_image,_= clickable_image.shape
        
        if len(clicks) == 0:
            yolo_format = ""
        
        else:
            yolo_format = cls._get_yolo_format(class_number,clicks[0],clicks[1],height_image,width_image)

            cls._visualization_annotation(image_path,clicks[0],clicks[1])

        return yolo_format
    
    @classmethod
    def _create_dataset_folders_and_yaml(cls,paths,yolo_dataset,DATA_FILE_NAME="data.yaml"):
        for path in paths:
            os.makedirs(path,exist_ok=True)
        
        data_content = {"path":yolo_dataset,"train":"images/train","val":"images/val","test":"images/test","names":{0:"person"}}
        data_path = os.path.join(yolo_dataset,DATA_FILE_NAME) 

        if os.path.exists(data_path):
            return

        with open(data_path,"w") as data_yaml_file:
            yaml.dump(data_content,data_yaml_file)
    
    @classmethod
    def _write_annotations(cls,data_element,dataset,item_name,yolo_dataset,LABELS="labels",IMAGES="images",TEXT_POSTFIX=".txt",
                           TRAIN_CONSTANT=8,VAL_CONSTANT=9,TEST_CONSTANT=0,PERCENTAGE_CONSTANT=10,TRAIN_NAME="train",VAL_NAME="val",TEST_NAME="test"):
        
        yolo_images_path = os.path.join(dataset,IMAGES)
        all_labels_path = os.path.join(dataset,LABELS)
        dataset_and_nested_paths = [dataset,yolo_images_path,all_labels_path,
                                    os.path.join(yolo_images_path,TRAIN_NAME),os.path.join(yolo_images_path,VAL_NAME),
                                    os.path.join(yolo_images_path,TEST_NAME),os.path.join(all_labels_path,TRAIN_NAME),
                                    os.path.join(all_labels_path,VAL_NAME),os.path.join(all_labels_path,TEST_NAME)]
        
        prefix_filename = os.path.basename(data_element.img_path).split(".")[0]
        folder_mode_name = ""

        cls._create_dataset_folders_and_yaml(dataset_and_nested_paths,yolo_dataset)

        # if not prefix_filename.isdigit():
        #     raise cls.WrongFileName(item_name)
        
        # index = int(prefix_filename) % PERCENTAGE_CONSTANT

        # if index <= TRAIN_CONSTANT and index != TEST_CONSTANT:
        #     folder_mode_name = TRAIN_NAME

        # elif index == VAL_CONSTANT:
        #     folder_mode_name = VAL_NAME

        # elif index == TEST_CONSTANT:
        
        folder_mode_name = TEST_NAME

        shutil.move(data_element.img_path,os.path.join(yolo_images_path,folder_mode_name))

        label_path_folder = os.path.join(all_labels_path,folder_mode_name,prefix_filename)
        label_file = f"{label_path_folder}{TEXT_POSTFIX}" 

        with open(label_file, "w") as yolo_label:
            yolo_label.write(data_element.yolo_format)

    @classmethod
    def iterate_images(cls,input_folder,yolo_dataset):
        
        if not os.path.isdir(input_folder):
            raise cls.WrongInputFolder(input_folder)

        class_number = os.path.basename(input_folder)
        
        for item_path in os.scandir(input_folder):
            item_name = os.path.basename(item_path)
            
            if not ( item_name.endswith(".png") or item_name.endswith(".jpg")):
                raise cls.FolderContainsWrongFile(item_name)
            
            yolo_Label = cls._anotate_file(class_number,item_path.path)
            
            data_element = cls.Data(yolo_Label,item_path.path)
        
            cls._write_annotations(data_element,yolo_dataset,item_name,yolo_dataset)

def main():
    args = parser.parse_args()

    ImageData.iterate_images(args.input_folder,args.yolo_dataset)

if __name__ == "__main__":
    main()
