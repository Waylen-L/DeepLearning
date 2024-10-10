import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


class RatingModel(nn.Module):
    def __init__(self):
        super(RatingModel, self).__init__()
        self.resnet = models.resnet50(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.resnet(x)


# Predict rating using the model
def predict_rating(image_path, model):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Predict
    model.eval()
    with torch.no_grad():
        predicted_rating = model(img_tensor)

    return predicted_rating.item()


# Predict the rating for a new image
new_image_path = 'val.jpg'  # Replace with the path to your image

# Load the model
model = RatingModel()
state_dict = torch.load('best_model.pth', weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.eval()

predicted_score = predict_rating(new_image_path, model)
print(f'Predicted score: {predicted_score}')