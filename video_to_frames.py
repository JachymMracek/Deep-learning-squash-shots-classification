import os
import argparse
import cv2

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--video_dataset_path",default = r"",help="Please, write path to video dataset")
argument_parser.add_argument("--output_frames_path",default = r"",help="Please, write output frame where the frames should be")

def main():
    
    args = argument_parser.parse_args()
    video_dataset_path = args.video_dataset_path
    output_frames_path = args.output_frames_path
    
    os.makedirs(output_frames_path, exist_ok=True)
    
    count = 0
    
    for _,video_name in enumerate(os.listdir(video_dataset_path)):
            video_path = os.path.join(video_dataset_path,video_name)
            
            video_capturer = cv2.VideoCapture(video_path)
        
            while True:
                
                readable, frame = video_capturer.read()
                
                if readable:
                    count += 1
                    
                    frame_path = os.path.join(output_frames_path,f"{count}.jpg")
                    print(frame_path)
                    cv2.imwrite(frame_path,frame)
                
                else:
                    break

if __name__ == "__main__":
    main()
