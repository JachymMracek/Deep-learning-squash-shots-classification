import os
import cv2
import argparse

# SOURCES:
# https://www.geeksforgeeks.org/python/python-create-video-using-multiple-images-using-opencv/

################################################################################
################################################################################
######################  Get TrackNet test videos with 3 frames #################
################################################################################
################################################################################

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--folder_triples",default = r"", help="Please,path to frames of triples")
argument_parser.add_argument("--video_output_folder",default = r"", help="Please, output path were videos of size three will be")

def create_video(triple_frames,video_output_folder,video_name):
    
    height, width, _ = triple_frames[0].shape
    
    video_path = os.path.join(video_output_folder,video_name)
    
    video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'DIVX'), 1, (width,height))
    
    for frame in triple_frames:
        video.write(frame)
    
    video.release()

"""
Creates videos of three frames from the folder of triple frames
"""

def iterate_folder(folder_with_frames,video_output_folder,TRIPLE = 3,VIDEO_FRAMES = 63,VIDEO_TEST_INDEX = 180):
    
    triple_frames = []
    video_index = VIDEO_TEST_INDEX
    triple_index = 1
    
    sorted_frame_names = sorted(os.listdir(folder_with_frames),key = lambda x:100*int(x.split("_")[0]) + int(x.split("_")[1].split(".")[0]))
    
    for i,frame_name in enumerate(sorted_frame_names):
        frame_path = os.path.join(folder_with_frames,frame_name)
        frame = cv2.imread(frame_path)
        print(frame_name)
        
        if (i+1) % TRIPLE == 0 and triple_frames == []:
            triple_frames.append(frame)
        
        elif (i+1) % TRIPLE == 0 and triple_frames != []:
            triple_frames.append(frame)
            create_video(triple_frames,video_output_folder,f"{video_index}_{triple_index}.mp4")
            triple_frames = []
            triple_index += 1
        
        elif (i+1) % TRIPLE != 0:
            triple_frames.append(frame)
        
        if triple_index > VIDEO_FRAMES / 3:
            video_index += 1
            triple_index = 1

def main():
    args = argument_parser.parse_args()
    
    folder_triples = args.folder_triples
    video_output_folder  = args.video_output_folder
    
    os.makedirs(video_output_folder,exist_ok=True)
    
    iterate_folder(folder_triples,video_output_folder)

if __name__ == "__main__":
    main()