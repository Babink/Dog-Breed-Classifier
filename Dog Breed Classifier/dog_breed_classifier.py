import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision.models as models
import random

# loading folder-name for human and dogs images
dog_files = np.array(glob('./dogImages/*/*/*'))
human_files = np.array(glob('./lfw/*/*'))
# print("There are %d images for dog dataset" % len(dog_files))
# print("There are %d images for Human dataset" % len(human_files))
# let's try implementing human detection
# we'll be using haar feature-based cascade classifier
# opencv provide us with pre-trained face detector , which is in frontal_face.xml file
# extracting pre-trained face detector
# now face_cascade can take parameters and predict whether it has face or not
face_cascade = cv2.CascadeClassifier('./frontal_face.xml')
# let's load color BRG images and convert it to grey scale image
# opencv don't really work with RBG images bbecause back in when technology was starting to rise , many camera manufacturer
# though that BRG will be a great choice
img = cv2.imread(human_files[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# let's find face in our image
# human face detected using pre-trained opencv model
faces = face_cascade.detectMultiScale(gray)
print("Number of face detected: {}".format(len(faces)))
# let's create a square around faces
# w , h denotes width and height of square box
# x , y is top left corner boundary
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


# let's convert our BRG into RBG for plotting
cv_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(cv_rbg)
# plt.show()


# let's create a function which will return true if there's any human face detected by pre-trained algorithms
def search_face(m_img):
    img = cv2.imread(m_img)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale)
    return len(faces) > 0

# let's check what percent of images is classified as Human for 100 dataset
# for both dog and human


human_short_files = human_files[:100]
dog_short_files = dog_files[:100]

# it's better to use float number than regular number in neural net algorithms
human_detected = 0.0
dog_detected = 0.0

for i in range(0, len(human_short_files)):
    human_path = human_short_files[i]
    dog_path = dog_short_files[i]

    if search_face(human_path) == True:
        human_detected += 1

    if search_face(dog_path) == True:
        dog_detected += 1

# print("Haar face detection")
# print("The percentage of the detected face in Human Dataset: {0:.0%}".format(
#     human_detected / len(human_short_files)))
# print("The percentage of the detected face in Dog Dataset: {0:.0%}".format(
#     dog_detected / len(dog_short_files)
# ))


# let's use torch to obtain pre-trained VGG-16 model
# code below will download the model along with weight that have been trained in ImageNet
VGG16 = models.vgg16(pretrained=True)
# above code will return a predication for the object tbat is contained in the images
# now let's create a model that will accept input and return index corresponding to the ImageNet class that is predicted
# by VGG16 model , Output will be from 0 - 999.

from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

human_new_files = human_files[:100]
dog_new_files = dog_files[:100]

def VGG16_predict(img_path):
    img = Image.open(img_path)

    # VGG16 takes 224*224 images as input
    # normalizing inputs images to make its elements from 0 to 1
    data_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std=[0.229 , 0.224 , 0.225])
        ]
    )

    # applying above transformation in our image
    img = data_transform(img)
    # torch pretrained models expect the tensor dimensions to be ie (no. of inputs , num color chaneels , height , widths)
    # currently , we have (num of color , height , width) , let's fix this by inserting a new axis
    # inserting new item at index 0 
    img = img.unsqueeze(0)
    # now our image is preprocessed , we need to convert it into a variable; pytorch models expect inputs to be variable
    # a torch variable is a wrapper around a torch tensor
    img = Variable(img)

    #returns a tensor  of shape (batch , num of class labels)
    predication = VGG16(img)
    predication = predication.data.numpy().argmax()
    # returning the index of the predicated class for that image
    return predication


# now let's create a dog_classifier 
def dog_classifier(dog_img):
    class_index = VGG16_predict(dog_img)
    
    if class_index >= 151 and class_index <= 268:
        return True
    else:
        return False



# let's predict 
# it's just a simple predication

simple_dog_detect = 0.0
simple_human_detect = 0.0

no_of_files = len(dog_new_files)

for i in range(0 , no_of_files):
    human_path = human_new_files[i]
    dog_path = dog_new_files[i]
    
    
    if dog_classifier(human_path) == True:
        simple_human_detect += 1
        
    if dog_classifier(dog_path) == True:
        simple_dog_detect += 1
        
print("VGG-16 Model Classifition")
print("The percentage of the detected dog in human face : {0:.0%}".format(simple_human_detect / no_of_files))
print("The percentage of the detected dog in dog face: {0:.0%}".format(simple_dog_detect / no_of_files))

# VGG-16 is best model for dog breed classifier 
# there are other best models like -> Inception-v3 , ResNet-50. Feel Free to try this model on your free time

# let's create our own cnn model to classify dog breed
# now we have functions that can predict human and dog faces
# it is hard to predict dog breed as many of the breed seem to have same facial, even we human can't interpret
# so it will be hard ;)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# check if cuda is available or not
use_cuda = torch.cuda.is_available()
print("Is Cuda Available: {}".format(use_cuda))

# let's transform our training , testing and validation dataset
# first resizing it cause VGG-16 model input size is 224 * 224
# convert it into tensor
# normalize imagge cause image pixel value should be between 0 - 1

# more details on transformation of datasets
# randomHorizontalFlip was used because we don't want our predication to be changed depends on the size,
# rotation or translation of images

from torchvision.transforms import transforms

transform = {
        'train': transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485 , 0.456 , 0.406],
                                         std=[0.229 , 0.224 , 0.225]
                                         )
                ]),
        'test': transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485 , 0.456 , 0.406] ,
                                       std=[0.229 , 0.224 , 0.225]
                                       )
                ]),
                  
        'valid': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485 , 0.456 , 0.406],
                                         std=[0.229 , 0.224 , 0.225]
                                         )
                ])
        }
        
        
# creating loader for each dataset
from torchvision import datasets
from torchvision import utils
import os


# numbers of subprocesser , if it's zero then use main processor
num_of_workers = 0
# how many samples will be loaded for one batch
batch_size = 10


# let's create our dataset
image_datasets = {
            x: datasets.ImageFolder(os.path.join('dogImages' , x) , transform[x])
            for x in ['train' , 'valid' , 'test']
        }

# now create our data loader
data_loader = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_of_workers
                                       ) 
        for x in ['train' , 'valid' , 'test']
            }
# decrease the batch size because of the out of memory in the GPU instance
test_loader = torch.utils.data.DataLoader(
            image_datasets['test'],
            shuffle=True,
            batch_size=15
        )



# checking our dataset information
data_sets = { x: len(image_datasets[x]) for x in ['train' , 'valid' , 'test']}

print("Number of record in training set: {}".format(data_sets["train"]))
print("Number of record in testing set: {}".format(data_sets["test"]))
print("Number of record in validation set: {}".format(data_sets["valid"]))

# let's check all the dog breed
breed_name = image_datasets["train"].classes
print(breed_name)
# number of classes
n_classes = len(breed_name)
print("There are total of {} classes".format(n_classes))
# displaying one record
# the images should be normalized , the labels is a integer value between 0-132
data_loader["train"].dataset[6679]

# visualizing images
def visualizing_img(img):
    img = img.numpy().transpose((1 , 2 , 0))
    img = np.clip(img , 0 , 1)
    
    plt.figure(figsize=(50 , 25))
    plt.axis('off')
    plt.imshow(img)
    plt.pause(0.001)
    
    
# get a batch of training data    
inputs , classes = next(iter(data_loader['train']))

# convert a batch to a grid
grid = utils.make_grid(inputs)

# display
visualizing_img(grid)



# now let's dive deeper into CNN
import torch.nn as nn

# defining cnn architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet , self).__init__()
        # size is 224
        self.conv1_1 = nn.Conv2d(3 , 64 , kernel_size=3 , padding=1)
        self.conv1_2 = nn.Conv2d(64 , 64 , kernel_size=3 , padding=1)
        
        # size is 112
        self.conv2_1 = nn.Conv2d(64 , 128 , kernel_size=3 , padding=1)
        self.conv2_2 = nn.Conv2d(128 , 128 , kernel_size=3 , padding=1)
        
        # size 56
        self.conv3_1 = nn.Conv2d(128 , 256 , kernel_size=3 , padding=1)
        self.conv3_2 = nn.Conv2d(256 , 256 , kernel_size=3 , padding=1)
        self.conv3_3 = nn.Conv2d(256 , 256 , kernel_size=3 , padding=1)
        # size 28
        self.conv4_1 = nn.Conv2d(256 , 512 , kernel_size=3 , padding=1)
        self.conv4_2 = nn.Conv2d(512 , 512 , kernel_size=3 , padding=1)
        self.conv4_3 = nn.Conv2d(512 , 512 , kernel_size=3 , padding=1)
        # size 14
        self.conv5_1 = nn.Conv2d(512 , 512 , kernel_size=3 , padding=1)
        self.conv5_2 = nn.Conv2d(512 , 512 , kernel_size=3 , padding=1)
        self.conv5_3 = nn.Conv2d(512 , 512 , kernel_size=3 , padding=1)
        
        # batch normalization is a technique to improve performance and stability of an ANN
        # it provide zero mean and unit variance as inputs to any layers
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.batch_norm256 = nn.BatchNorm2d(256)
        self.batch_norm512 = nn.BatchNorm2d(512)
        
        
        # max_pooling is used to reduce the size of images and the amount of parameters in half and 
        # to capture the most useful pixel
        self.max_pool = nn.MaxPool2d(kernel_size=2 , stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout() # For better weight and bias updating
        
        # size 7
        # Fully connected layer
        self.fc1 = nn.Linear(512 * 7 * 7 , 4096)
        self.fc2 = nn.Linear(4096 , 4096)
        
        # last output of fully connected layer is our result which will be decided by relu activation func (133)
        self.fc3 = nn.Linear(4096 , 133)
        
        
    def forward(self , x):
        x = self.relu(self.batch_norm64(self.conv1_1(x)))
        x = self.relu(self.batch_norm64(self.conv1_2(x)))
        x = self.max_pool(x)
        
        x = self.relu(self.batch_norm128(self.conv2_1(x)))
        x = self.relu(self.batch_norm128(self.conv2_2(x)))
        x = self.max_pool(x)
        
        x = self.relu(self.batch_norm256(self.conv3_1(x)))
        x = self.relu(self.batch_norm256(self.conv3_2(x)))
        x = self.relu(self.batch_norm256(self.conv3_3(x)))
        x = self.max_pool(x)
        
        x = self.relu(self.batch_norm512(self.conv4_1(x)))
        x = self.relu(self.batch_norm512(self.conv4_2(x)))
        x = self.relu(self.batch_norm512(self.conv4_3(x)))
        x = self.max_pool(x)
        
        x = self.relu(self.batch_norm512(self.conv5_1(x)))
        x = self.relu(self.batch_norm512(self.conv5_2(x)))
        x = self.relu(self.batch_norm512(self.conv5_3(x)))
        x = self.max_pool(x)
        
        # now x return a new tensor which has a different size
        # - 1 means inferring (conclude) the size from other dimensions.
        x = x.view(x.size(0) , -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
# let's create CNN instance
CNN_Classifier = NeuralNet()

# if cuda is available , move tensors to GPU
if use_cuda:
    CNN_Classifier.cuda()
        

# specifying loss functions and optimizer
import torch.optim as optim
#let's select our loss functions
criterion = nn.CrossEntropyLoss()
# selecting our optimizer
optimizer = optim.SGD(CNN_Classifier.parameters() , lr=0.001 , momentum=0.9)

# let's train and validate our model
def train(epochs , train_loader , valid_loader , model , optimizer , criterion , use_cuda , save_path):
    # return trained model
    # initializer tracker for minimum validation loss
    valid_loss_min = np.Inf
    
    
    for epochs in np.arange(1 , epochs+1):
        # initalize variable to calculate loss
        train_loss = 0.0
        valid_loss = 0.0
        
        for batch_idx , (data , target) in enumerate(train_loader):
            if use_cuda:
                data , target = data.cuda() , target.cuda()
                
                # todo:
                # -> find the loss and update the model parameters 
                # -> Record average training loss 
                # clear the gradient of all optimized variable
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output , target)
                loss.backward()
                optimizer.step()
                
                # updating training loss
                train_loss += loss.item() * data.size(0)
                
        # work
        # do same for validation_loss
        # calculate average loss
        # print training/validation stats
        # save if validation loss has decrease
            # compare with valid_loss_min , and if it is higher then valid_loss save model
        
        
        for batch_idx , (data , target) in enumerate(valid_loader):
            if use_cuda:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output , target)
                loss.backward()
                optimizer.step()
                
                valid_loss += loss.item() * data.size(0)
                
        
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        
        print("Epochs:{} , Training Loss: {} , Validation Loss: {} ".format(epochs , train_loss , valid_loss))
        
        if valid_loss <= valid_loss_min:
            print("Validation Loss decreased {{:.6f}} --> {:.6f}.   Saving Model......"
                  .format(valid_loss_min , valid_loss)
                  )
            
            torch.save(model.state_dict() , save_path)
            valid_loss_min = valid_loss
            
            
    return model        
            

epochs = 15
# training model
model_scratch = train(epochs , data_loader['train'] , data_loader['valid'], 
    CNN_Classifier , optimizer , criterion , use_cuda , "model.pt")                
# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model.pt'))


# testing model
# let's apply test data into our trained model
# code below will calculate accuracy and loss , our goal is to get accuracy greater than 10%

def test_model(loader , model , criterion , use_cuda):
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    
    for batch_idx , (data , target) in enumerate(loader):
        if use_cuda:
            data , target = data.cuda() , target.cuda()
        
        
        # forward pass: compute predicated outputs by passing inputs to the model
        output = model(data)
        # calculate loss
        loss = criterion(output , target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # converting output probability into predicated class
        pred = output.data.max(1 , keepdim=True)[1]
        
        # compare predication to it's corresponding targets / labels
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        
        print("Test loss => {:.6f}\n".format(test_loss))
        print("\n Test accuracy: %2d%% (%2d/%2d)" % (100.0 * correct / total , correct , total))


test_model(test_loader , CNN_Classifier , criterion , use_cuda)


# use transfer learning to transfer our pre-trained cnn model to classify dog breed
# loading vgg-16 model
model_transfer = models.vgg16(pretrained=True)

# freeze pre-trained weight
for param in model_transfer.features.parameters():
    param.required_grad = False

# get the input of the last layer of vgg-16
n_inputs = model_transfer.classifier[6].in_features

# create a new layer (n_inputs -> 133)
# new layer required_grad will be True automatically
last_layer = nn.Linear(n_inputs , 133)

# change last layer into new layer
model_transfer.classifier[6] = last_layer

print(model_transfer)

if use_cuda:
    model_transfer = model_transfer.cuda()


criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.paramters() , lr=0.001)


# let's train our transfer model
n_epochs = 20
model_transfer = train(n_epochs,
                       data_loader['train'],
                       data_loader['valid'],
                       model_transfer,
                       optimizer_transfer,
                       criterion_transfer,
                       use_cuda,
                       'model_transfer.pt'
                       )

# load the model that got the best validation accuracy 
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


# testing model
test_model(data_loader['test'] , model_transfer , criterion_transfer , use_cuda)

# predicting dog breed with our model
device = torch.device("cuda:0" if use_cuda else "cpu")

# load the trained model 'model_transfer.pt'
model_transfer.load_state_dict(torch.load('model_transfer.pt' , map_location='cpu'))

import torchvision.transforms as transforms

def predict_dog_breed(img_path):
    # class name with number -> 0001.golden retreivial
    # class name without number -> golden retreivial
    class_name_with_number = image_datasets['train'].classes
    class_name_without_number = [item[4:].replace("_" , " ") for item in image_datasets['train']
        .classes
    ]
    
    # load image
    load_img = Image.open(img_path)
    # process image
    
    transform_predict = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                                    mean=[0.485 , 0.456 , 0.406],
                                    std=[0.229 , 0.224 , 0.225]
                                     )
            ])
                
    # get tensors
    img_tensor = transform_predict(load_img)
    img_tensor = img_tensor.unsqueeze_(0)
    # send tensor to device (gpu or cpu)
    img_tensor = img_tensor.to(device)
    
    # pytorch expect inputs to be a Variables
    img_var = Variable(img_tensor)
    
    output = model_transform(img_var)
    
    # getting probability of breeds
    softmax = nn.Softmax(dim=1)
    preds = softmax(output)
    
    # get three breed that have highest probability
    top_preds = torch.topl(preds , 3)

    # get the name of breeds just to display
    labels_without_number = [class_name_without_number[i] for i in top_preds[1][0]]
    labels_with_number = [class_name_with_number[i] for i in top_preds[1][0]]
    
    probs = top_preds[0][0]
    
    return labels_without_number , labels_with_number , probs



# now write an algorithms that will accept a file path to an image 
# and determine the image contains a human , dog or neither 
    
# first display image 
def display_image(img_path):
    img = Image.open(img_path)
    _ , ax = plt.subplot()
    ax.imshow(img)
    plt.axis('off')
    plt.show()

# display dog breed image
def display_dog_breed_image(labels):
    fig = plt.figure(figsize=(16 , 4))
    
    for i , labels in enumerate(labels):
        subdir = ''.join(['dogImages/valid/' , labels + '/'])
        file = random.choice(os.listdir(subdir))
        path = ''.join([subdir , file])
        img = Image.open(path)
        ax = fig.add_subplot(1 , 3 , i+1)
        ax.imshow(img , cmap='gray' , interpolation='nearest')
        plt.title(labels.split('.')[1])
        plt.axis('off')
        
    plt.show()
    
    
# now final step , running app
def run_app(img_path):
    # get the probabilities and labels
    labels_without_numbers , labels_with_numbers , probs = predict_dog_breed(img_path)
    
    # if it's dog
    if probs[0] > 0.3:
        # display the input image
        print("It's a dog")
        display_image(img_path)
        
        # displaying it's predicated breeds and it's probabilities
        print("Predicated Breeds and it's probability is: \n")
        for pred_labels , prob in zip(labels_without_numbers , probs):
            print(pred_labels)
            print('{:.2f}%'.format(100 * prob))
        print("\n")
        
        # displaying predicated breed image
        display_dog_breed_image(labels_with_numbers)
        
    elif search_face(img_path):
        print("It's HUMAN")
        display_image(img_path)
        
        # displaying most resembled breeds with that human face and it's probability
        print("Resembled Breeds and it's probability \n")
        for pred_labels , prob in zip(labels_without_numbers , probs):
            print(pred_labels)
            print('{:.2f}%'.format(100*prob))
        print('\n')
        
        
        display_dog_breed_image(labels_with_number)
    else:
        # not a human and dog
        print("Can't detect whether it is human or dog")
        display_image(img_path)
        print('\n')
        
        
# final testing
for file in np.hstack((human_files[:5]) , dog_files[:5]):
    run_app(file)
    
# .