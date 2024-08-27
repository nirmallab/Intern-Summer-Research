
import torch #gpu acceleration
device = "cpu"
if (torch.cuda.is_available()):
  device = "cuda"
print("device: " + device)

import h5py
def load_data(file_path): #function to read in labels and images (as numpy arrays from) the h5 file
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]
    return images, labels #return the images and labels

import pandas as pd
import os
# Load the array and labels from the HDF5 file
images, labels = load_data('medium.h5') #change from medium.h5 to whatever the name of the h5 file is

#putting all the labels and images into a dataframe for ease of access
data = pd.DataFrame()

data['images'] = list(images)
data['labels'] = labels

#split dataframe into test, train, and validation (10, 80, 10)
from sklearn.model_selection import train_test_split
train_data, rest_data = train_test_split(data, train_size=0.80, shuffle=True, random_state=10)
validation_data, test_data = train_test_split(rest_data, test_size=0.5, shuffle=True, random_state=10)

train_data = train_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


import os
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

#Custom Dataset class to utilize Dataframes for training pipeline 
class HEStainDataset(Dataset):
  def __init__(self, df, transform = -1):

    self.transform = transform #transformations to be applied to the images
    self.df = df #dataframe containing the images and labels
    
  def __len__(self):
    return len(self.df) #return length of dataframe

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
            
    image = self.df['images'][idx] #get the image at the index
    
    image = Image.fromarray(image) #convert the image to PIL format for applying transformations to it
    assert(image.mode == 'RGB') #assert that the image is in RGB format
    
    if (self.transform != -1):
      image = self.transform(image) #apply the transformations to the image

    #getting label
    label = torch.tensor(self.df['labels'][idx], dtype=torch.long)

    #warning is printed if removed
    label = label.clone().detach()
    
    #need to return in this dictionary form to be passed into huggingface transformer
    return {
        "pixel_values": image,
        "labels": label
    }
    

from torchvision import transforms

#The transformation we do for all datasets
commonTransform = transforms.Compose([
    transforms.Resize((224, 224)), #may change depending on model
    transforms.ToTensor(),
    transforms.Normalize([0.7756, 0.7037, 0.8011], [0.1126, 0.1126, 0.1126]) #gained from previous pyfiles code
])

# Data augmentation transformations
augmentTransform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation(30)], p=0.5), #50% chace for random 30 degree rotation
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.5) #50% chace for random pixel changes
])

#can add more transformations for throughness and preventing of overfitting

# The training transformation is a combination of the two previous transformations
trainTransform = transforms.Compose([
    augmentTransform,
    commonTransform
])

#create the train, test, and validation datasets
trainDataset = HEStainDataset(df = train_data, transform = trainTransform)

valDataset = HEStainDataset(df = validation_data, transform = commonTransform)

testDataset = HEStainDataset(df = test_data, transform = commonTransform)

from transformers import ViTForImageClassification, TrainingArguments, Trainer

model_name = "google/vit-base-patch16-224-in21k" #model architecture name

n_classes = 13 #number of classes

# Load the ViT model from directory where it is stored
model = ViTForImageClassification.from_pretrained("./vit-13") 

import torch.nn as nn

for param in model.parameters(): #freeze all previous layers
            param.requires_grad = False

#add a new linear classifier layer at the end
model.classifier = nn.Sequential(
        nn.Linear(768, 384),
        nn.ReLU(),
        nn.Linear(384, n_classes)
        )

from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

#function for computing accuracy and f1
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    f1 = f1_score(labels, preds, average='weighted')  #can change how f1 is calculated to whatever you see fit
    accuracy = accuracy_score(labels, preds)
    
    return {
        'f1': f1,
        'accuracy': accuracy
    }

from collections import defaultdict
from transformers.trainer_callback import TrainerCallback
import math

#Custom Callback class (how to store results)
class CustomCallback(TrainerCallback):
    def __init__(self):
      #where to store meterics
        self.valid_metrics = defaultdict(list) 
        self.train_metrics = defaultdict(list)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
          #How I kept track if the results from the outputted evaluation were from the validation or training (not an ideal solution) 
            if (state.epoch == math.trunc(state.epoch)): #the training results were outputted every x amount of steps which wasn't always an integer epoch number while the validation results were always at the end of the epoch
                #for validation, keep track of epoch, loss, f1, and accuracy
                self.valid_metrics['epoch'].append(state.epoch)
                self.valid_metrics['eval_loss'].append(logs.get('eval_loss', None))
                self.valid_metrics['eval_f1'].append(logs.get('eval_f1', None))
                self.valid_metrics['eval_accuracy'].append(logs.get('eval_accuracy', None))
                # print(f"Epoch: {state.epoch}, Eval Loss: {logs.get('eval_loss', None)}, Eval F1: {logs.get('eval_f1', None)}, Eval Accuracy: {logs.get('eval_accuracy', None)}")
            else: 
              #for training, only keep track of epoch and loss
                self.train_metrics['epoch'].append(state.epoch)
                self.train_metrics['train_loss'].append(logs.get('loss', None))
                # print(f"Epoch: {state.epoch}, Train Loss: {logs.get('loss', None)}")

    def get_metrics(self):
        return self.valid_metrics, self.train_metrics
    

callback = CustomCallback()

num_epochs = 50 #number of epochs

training_args = TrainingArguments(
    output_dir="./vit-13/outputs", #directory for outputs
    num_train_epochs=num_epochs, #set number of epochs
    per_device_train_batch_size=64, #set batch size
    learning_rate=1e-4, #set learning rate
    #weight_decay=0.001, # Applying weight decay for L2 regularization
    evaluation_strategy="epoch",  # Ensure validation evaluation occurs at the end every epoch
    save_strategy="epoch", #The the model will save a checkpoint at the end of each epoch. 
    load_best_model_at_end=True,  # Load best model found during training
    metric_for_best_model="f1",  # Use F1 score to determine the best model
    logging_steps=50, #change depending on speed of gpu/gpus
    save_total_limit=1 #save only the top model
)

#create trainer with arguments, datasets, and callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainDataset,
    eval_dataset=valDataset,
    compute_metrics=compute_metrics,
    callbacks=[callback]
)

#train
trainer.train()

#validation and training data
valid_met, train_met = callback.get_metrics()

#save validation data
validation_data = pd.DataFrame.from_dict(valid_met)
validation_data.to_csv("./validation_data.csv", index=False)

#save training data
training_data = pd.DataFrame.from_dict(train_met)
training_data.to_csv("./training_data.csv", index=False)

#test
results = trainer.evaluate(testDataset)

#remove unessacary information for test data
results.pop("eval_runtime")
results.pop("eval_samples_per_second")
results.pop("eval_steps_per_second")
results.pop("epoch")

# change names to reflect test data
results["test_accuracy"] = [(results.pop("eval_accuracy"))]
results["test_f1"] = [(results.pop("eval_f1"))]
results["test_loss"] = [(results.pop("eval_loss"))]

#save testing data
testing_data = pd.DataFrame.from_dict(results)
testing_data.to_csv("./testing_data.csv", index=False) 

print("done")
