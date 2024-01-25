
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model.inceptionresnet import InceptionResNet
import os

#add data to module path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_model(weights_path):
    model = InceptionResNet(num_classes=3)
    model.load_state_dict(torch.load(weights_path))
    return model

def load_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

def infer(model, image):
    output = model(image)
   
    #apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    print("probabilities:",probabilities)
    #get max probability
    _, predicted = torch.max(probabilities, 1)

    return predicted

def plot_image(image, image_path, prediction):
    plt.imshow(image.squeeze(0).permute(1, 2, 0))
    plt.text(5, 5, f'Path: {image_path}\nPrediction: {prediction}', bbox=dict(facecolor='red', alpha=0.5))
    plt.show()

def test_model(weights_path):
    image_paths = ['data/4- Modèle Transfert glissé/2- Carrelage/Retouches/16.jpeg','data/1- Modèle Traces passives/2- Carrelage/Carrelage NH/Retouches/7.jpeg']  # replace with your image paths

    model = load_model(weights_path)
    model.eval()

    #define the correspondence between class labels and class indices
    class_labels = ['1- Modèle Traces passives', "15- Modèle Modèle d'impact", '4- Modèle Transfert glissé']

    for image_path in image_paths:
        image = load_image(image_path)
        prediction = infer(model, image)
        print(f'Prediction for {image_path}: {prediction.item()}')
        
        print("This images corresponds to the class:",class_labels[prediction.item()])
        plot_image(image, image_path, class_labels[prediction.item()])

if __name__ == '__main__':

    #class_labels: ['1- Modèle Traces passives', "15- Modèle Modèle d'impact", '4- Modèle Transfert glissé'] 
    #Donc: - 0 est le modèle de traces passives
    #     - 1 est le modèle d'impact
    #     - 2 est le modèle de transfert glissé

    model_path = 'logs/resnet18_1/checkpoint.pt'
    test_model(model_path)

