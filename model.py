import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

# network class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.full1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        self.full2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(84, 10),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.full1(x)
        x = self.full2(x)

        return self.out(x)

    def predict_(self, images):
        test_output = self(images)
        print(test_output)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        return pred_y, test_output

    def predict_img(self, img):
        pred, pososta = self.predict_(ToTensor()(img).resize(1, 1, 28, 28))
        return pred.item(), pososta[0]


def to_grayscale_28(img):
    return img.convert('L').resize((28, 28))

def load_model():
    model = CNN()
    model.load_state_dict(torch.load("model_cnn.data"))
    model.eval()

    return model