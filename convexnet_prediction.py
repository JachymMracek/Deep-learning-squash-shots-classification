from torchvision import transforms
import cv2
import torch
from PIL import Image
import argparse

# https://discuss.pytorch.org/t/torch-max-and-softmax-confusion/80697

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--convexnet_weights_path",default = r"D:\FOLDER_OF_FINAL_BACHELOR_THESIS\ball_hit_model.pth",help="Please, write path to the convNeXt model")
argument_parser.add_argument("--frame_path",default = r"C:\Users\jachy\Documents\ShareX\Screenshots\2026-07\shot_comparision.png",help="Please, write path to the frame")

def convexnet_prediction(frame,convexnet_classificator):
    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    test_trasformers = transforms.Compose([transforms.Resize(224),transforms.CenterCrop((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    
    # https://www.geeksforgeeks.org/python/convert-opencv-image-to-pil-image-in-python/
    
    frame_change_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_change_color)
    shot_tensor = test_trasformers(frame_pil)
    shot_tensor = shot_tensor.unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        outputs = convexnet_classificator(shot_tensor)
        confidencies = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predictions = torch.max(confidencies, 1)
    
    class_predicted = predictions.tolist()[0]
    
    return class_predicted,confidence.tolist()[0]

def main():
    args = argument_parser.parse_args()
    
    convexnet_weights_path = args.convexnet_weights_path
    frame_path = args.frame_path
    
    convexnet_classificator = torch.load(convexnet_weights_path,weights_only=False)
    frame = cv2.imread(frame_path)
    
    class_predicted,confidence = convexnet_prediction(frame,convexnet_classificator)
    
    print("class_predicted: " + f"{class_predicted}","confidence: " + f"{confidence}")

if __name__ == "__main__":
    main()