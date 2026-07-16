import torch
import argparse
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score,accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################
################################################################################
############################ get convNexT performance ##########################
################################################################################
################################################################################

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--folder_of_hit_images",default =r"",help="Please, write path to folder with shot images")
argument_parser.add_argument("--convexnet_weights",default = r"",help="Please, write path to the convNeXt model weights")
argument_parser.add_argument("--classes",default=["boast", "cross", "drive", "drop"],help='["boast", "cross", "drive", "drop"] or ["not hit", "hit"]')

def eval_dataset(phase_dataloader,model,classes):
    
    all_predictions = []
    all_labels = []
    
    for data,labels in phase_dataloader:
        data = data.to(device)
        labels = labels.to(device)
            
        outputs = model(data)
        _, predictions = torch.max(outputs, 1)
        
        all_predictions.append(predictions.to(device))
        all_labels.append(labels.data)
    
    all_predictions = torch.cat(all_predictions,dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels,dim=0).cpu().numpy()
    
    precission = precision_score(all_labels,all_predictions,average="macro")
    recall = recall_score(all_labels,all_predictions,average="macro")
    f1 = f1_score(all_labels,all_predictions,average="macro")
    accuracy = accuracy_score(all_labels,all_predictions)
    
    ConfusionMatrixDisplay.from_predictions(all_labels,all_predictions,display_labels=classes)
    
    plt.show()
    
    classes_precision = precision_score(all_labels,all_predictions,average=None)
    classes_recall = recall_score(all_labels,all_predictions,average=None)
    classes_f1 = f1_score(all_labels,all_predictions,average=None)
    
    print("precision for each class: " + f"{classes_precision}")
    print("recall for each class: " + f"{classes_recall}")
    print("f1 for each class: " + f"{classes_f1}")
    print("Dataset precision: " + f"{precission}" + " " " Dataset recall: " +  f"{recall}" + " Dataset f1: " + f"{f1}" + " Dataset accuracy: " + f"{accuracy}")

def main(TRANFORMATOR = transforms.Compose([transforms.Resize(224),transforms.CenterCrop((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),BATCH_SIZE = 8,NUM_WORKERS = 8):
    
    args = argument_parser.parse_args()
    folder_of_hit_images = args.folder_of_hit_images
    convexnet_model_weights = args.convexnet_weights
    classes = args.classes
    
    model = torch.load(convexnet_model_weights,weights_only=False)
    
    dataset = torchvision.datasets.ImageFolder(root=folder_of_hit_images,transform =TRANFORMATOR)
            
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    
    eval_dataset(dataloader,model,classes)

if __name__ == "__main__":
    main()