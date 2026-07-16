import torch
import torchvision
from torchvision import models,transforms
import torch.nn as nn
import argparse
from enum import Enum
import copy
import os
import torch.optim as optim
from torchvision.models import ConvNeXt_Tiny_Weights
import matplotlib.pyplot as plt
import utils

# https://docs.pytorch.org/vision/stable/models.html
# https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/finetuning_torchvision_models_tutorial.ipynb#scrollTo=0WEANIHZhstz
# https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/
# https://www.geeksforgeeks.org/python/abstract-classes-in-python/
# https://docs.python.org/3/library/enum.html
# https://www.w3schools.com/python/ref_random_seed.asp
# https://www.geeksforgeeks.org/python/graph-plotting-in-python-set-1/
# https://medium.com/we-talk-data/how-to-set-random-seeds-in-pytorch-and-tensorflow-89c5f8e80ce4
# https://www.appsilon.com/post/transfer-learning-introduction
# https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
# https://apxml.com/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/data-transformations-torchvision-transforms

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--root_image_folder",default=r"",
                             help = "root folder path to folder for classification where are splited train/val/test frames")
        
INPUT_SIZE = 224
SEED = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class Classificator:
    
    """
    In this class ConvNeXt is trained
    """
    
    # enum for phases. 
    class Phase(Enum):
        TRAIN = 0
        VAL = 1
        TEST = 2
    
    
    """
    get dataloader for training or validating
        
    Arguments:
        folder_image_phase_path (string): path to phase folder with images
        transformator (torchvision.transforms.Compose): transformator for data in current phase.
    
    Returns:
        dataloader for current phase (torch.utils.data.DataLoader)
        
    """
    
    @classmethod
    def get_dataloader(cls,folder_image_phase_path,transformator,BATCH_SIZE = 8,NUM_WORKERS = 8):
        dataset = torchvision.datasets.ImageFolder(root=folder_image_phase_path,transform = transformator)
            
        return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    
        
    """
    draw training curves after training
        
    Arguments:
        train_data (list[int]): list of training values for plottting. In general they are values of losses or accuracy.
        val_data (list[int]): list of validation values for plottting. In general they are values of losses or accuracy.
    """
    
    @classmethod
    def draw_training_curves(cls,train_data,val_data,epochs,train_title,val_title,y_label,X_LABEL = "epoch"):
        plt.plot(epochs,train_data,color= "blue",label = train_title)
        plt.plot(epochs,val_data,color = "orange",label = val_title)
        
        plt.xlabel(X_LABEL)
        plt.ylabel(y_label)
        
        plt.legend()
        plt.show()
    
    """
    get model for the training
        
    Arguments:
        number_of_classes (int): number of classes for fine tuning
    
    Returns:
        model (torchvision.models): model for training already on device
    """

    @classmethod
    def get_model(cls,number_of_classes):
        convexnet_tiny_model = models.convnext_tiny(weights = ConvNeXt_Tiny_Weights.DEFAULT)
        
        number_of_current_classes = convexnet_tiny_model.classifier[2].in_features
        convexnet_tiny_model.classifier[2] = nn.Linear(number_of_current_classes,number_of_classes)   
        
        return convexnet_tiny_model.to(device)
    
    """
    Evaluates current folder for output message during training
        
    Arguments:
        phase_dataloader (torch.utils.data.DataLoader): dataloader where are phase data used
        optimizer (torch.optim.Optimizer): optimizer of the evalution
        phase_idx (int): index of the phase. train has 0 and val has 1
        model (torchvision.models): models which is used for evaluation
        criterion (torch.nn.Module): criterion of the evalution.
        
    """

    @classmethod
    def evaluate_folder(cls,phase_dataloader,optimizer,phase_idx,model,criterion):
        current_loss = 0
        current_correct_predictions = 0
        
        i = 0
        for data,labels in phase_dataloader:
            i += 1
            data = data.to(device)
            labels = labels.to(device)
                
            optimizer.zero_grad()
                    
            with torch.set_grad_enabled(phase_idx == cls.Phase.TRAIN.value):
                outputs = model(data)
                loss = criterion(outputs, labels)
                _, predictions = torch.max(outputs, 1)
                        
                if phase_idx == cls.Phase.TRAIN.value:
                    loss.backward()
                    optimizer.step()
                    
                current_loss += loss.item() * data.size(0)
                current_correct_predictions += torch.sum(predictions == labels.data)
                    
            current_accuracy = 100*float(current_correct_predictions / len(phase_dataloader.dataset))
            current_loss = float(current_loss)
        
        return current_loss, current_accuracy

        
    """
    train classificator
        
    Arguments:
        model (torchvision.models): models which is trained
        dataloaders (list[torch.utils.data.DataLoader]): list of of training and validation frames.
        optimizer (torch.optim.Optimizer): optimizer of the training.
        criterion (torch.nn.Module): criterion of the training.
    """
    
    @classmethod
    def train(cls,model,dataloaders,optimizer,criterion,EPOCHS =30,COUNT_PHASES = 2,
            
            LOSS_TITLE = "loss",
            ACCURACY_TITLE = "accuracy",
            VAL_LOSS_TITLE = "val loss",
            TRAIN_LOSS_TITLE = "train loss",
            VAL_ACCURACY_TITLE = "val accuracy",
            TRAIN_ACCURACY_TITLE = "train accuracy",
            MODEL_NAME = "all_real.pth"):
        
        best_model_weights = None
        best_accuracy_model = float("-inf")
        
        train_accuracy = []
        val_accuracy = []
        train_loss = []
        val_loss = []
        
        for epoch in range(EPOCHS):
            for phase_idx in range(COUNT_PHASES):
                print(epoch)
                
                if phase_idx == cls.Phase.TRAIN.value:
                    model.train()
                
                elif phase_idx == cls.Phase.VAL.value:
                    model.eval()
                
                phase_dataloader = dataloaders[phase_idx]
                
                current_loss, current_accuracy = cls.evaluate_folder(phase_dataloader,optimizer,phase_idx,model,criterion)
            
                if phase_idx == cls.Phase.TRAIN.value:
                    train_loss.append(current_loss)
                    
                    train_accuracy.append(current_accuracy)
            
                elif phase_idx == cls.Phase.VAL.value:
                    val_loss.append(current_loss)
                    val_accuracy.append(current_accuracy)
                
                if phase_idx == cls.Phase.VAL.value and current_accuracy > best_accuracy_model:
                    best_accuracy_model = current_accuracy
                    best_model_weights = copy.deepcopy(model.state_dict())
                                    
                    model.load_state_dict(best_model_weights)
                    torch.save(model,MODEL_NAME)
        
        epochs_idx = [i+1 for i in range(EPOCHS)]
        
        cls.draw_training_curves(train_loss,val_loss,epochs_idx,TRAIN_LOSS_TITLE,VAL_LOSS_TITLE,LOSS_TITLE)
        cls.draw_training_curves(train_accuracy,val_accuracy,epochs_idx,TRAIN_ACCURACY_TITLE,VAL_ACCURACY_TITLE,ACCURACY_TITLE)
    
    """
    prepare dataset for training
        
    Arguments:
        root_folder (string): path to root folder where are nested folders with class images
    """
    
    @classmethod
    def prepare_for_training(cls,root_folder,
        
        TRAIN_TRANFORMATOR = transforms.Compose([transforms.RandomRotation(10),transforms.RandomHorizontalFlip(p=0.5),transforms.Resize(224),transforms.CenterCrop((224, 224)),transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
           
        VALIDATION_TRANFORMATOR = transforms.Compose([transforms.Resize(224),transforms.CenterCrop((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
    
        TRAIN_FOLDER_NAME = "train",
    
        VALIDATION_FOLDER_NAME = "val"
    ):
            
        train_folder_path = os.path.join(root_folder,TRAIN_FOLDER_NAME)
        val_folder_path = os.path.join(root_folder,VALIDATION_FOLDER_NAME)
        
        train_dataloader = cls.get_dataloader(train_folder_path,TRAIN_TRANFORMATOR)
        val_dataloader = cls.get_dataloader(val_folder_path,VALIDATION_TRANFORMATOR)
        
        number_of_classes = utils.get_number_of_videos(train_folder_path)
        
        model = cls.get_model(number_of_classes)
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(model.parameters(),lr = 0.00001,betas = (0.9, 0.999), eps = 1e-6,weight_decay = 0, amsgrad = False)
        
        cls.train(model,[train_dataloader,val_dataloader],optimizer,criterion)

def main():
    
    args = argument_parser.parse_args()
    
    root_folder = args.root_image_folder
    
    Classificator.prepare_for_training(root_folder)

if __name__ == "__main__":
    main()
