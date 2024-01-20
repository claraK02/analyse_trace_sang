import torch
import torchvision.models as models
import torch.nn as nn

class InceptionResNet:
    def __init__(self, num_classes=10, pretrained=True, dropout_probability=0.5):   
        # Load the pre-trained ResNet-18 model
        self.model = models.inception_v3(pretrained=pretrained) #Ceci

        # Freeze all the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the last layer of the model
        # Add a dropout layer and modify the last layer of the model
        self.model.dropout = nn.Dropout(dropout_probability)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

if __name__ == '__main__':
    model = InceptionResNet()
    print(model.model)
    #test of the model
    #creata a random tensor
    x = torch.randn(1,3,512,512)
    #forward
    y = model.forward(x)
    # Apply softmax to the logits
    probabilities = torch.nn.functional.softmax(y.logits, dim=1) #que pour inception !

    # Get the predicted class
    _, predicted_class = torch.max(probabilities, 1)

    print("Classes:",predicted_class) #
    #print the output shape
    #print(y.shape)
    #print the output
    #print(y)