import os 
import shutil
import cv2
from annotator import ImageData
import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dataset_of_triples_path",default = r"",help="Please, write path to video dataset of triples")
argument_parser.add_argument("--last_triple_frame_dataset_path",default = r"", help="Please, write output path where last frames will be saved")

def anotate_last_frame_triples(dataset_path,last_triple_frame_dataset_path,LAST_FRAME_FROM_TRIPLE_FOLDER_NAME = "last_triple_frames"):

    root_folder = os.path.dirname(dataset_path)
    
    folder_of_last_frame_triples = os.path.join(root_folder,LAST_FRAME_FROM_TRIPLE_FOLDER_NAME)
    
    os.makedirs(folder_of_last_frame_triples,exist_ok=True)
    
    for frame_name in os.listdir(dataset_path):
        frame_video_index = int(frame_name.split("_")[1].split(".")[0])
        frame_path = os.path.join(dataset_path,frame_name)
        
        if frame_video_index % 3 == 0:
            frame_path_to_save = os.path.join(folder_of_last_frame_triples,frame_name)
            
            last_frame = cv2.imread(frame_path)
            
            cv2.imwrite(frame_path_to_save,last_frame)
    
    ImageData.iterate_images(folder_of_last_frame_triples,last_triple_frame_dataset_path)
    
    shutil.rmtree(folder_of_last_frame_triples)
    
def main():
    args = argument_parser.parse_args()
    
    dataset_of_triples_path = args.dataset_of_triples_path
    last_triple_frame_dataset_path = args.last_triple_frame_dataset_path
    
    os.makedirs(last_triple_frame_dataset_path,exist_ok=True)
    
    anotate_last_frame_triples(dataset_of_triples_path,last_triple_frame_dataset_path)

if __name__ == "__main__":
    main()
