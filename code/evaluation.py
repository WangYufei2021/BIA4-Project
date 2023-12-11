import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from torchvision.models import resnet50, densenet121
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc


# Define the Net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = densenet121()
        self.resnet.fc = nn.Linear(2048, 2)

    def forward(self, x):
        return self.resnet(x)


def draw_roc(true_labels, predicted_probs):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr,
             lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC curve')

    plt.legend(loc="lower right")
    plt.show()


# Path to the saved model
model_path = '/Users/A.BC.Perroquet/Library/CloudStorage/OneDrive-InternationalCampus,ZhejiangUniversity/ZJU/Fourth Year/BIA4/ICAs/ICA1/Model and dataset/Model and dataset/New model/densenet121_epoch100_97.61_best.pth'

# Load the model state dictionary
model = torch.load(model_path, map_location='cpu')
# print(model)
# Instantiate the model
# model = Net()
# model.load(model_state_dict)
model.eval()
# Define the transformation for the test images
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Path to the directory containing test images
benign_dir = '/Users/A.BC.Perroquet/Library/CloudStorage/OneDrive-InternationalCampus,ZhejiangUniversity/ZJU/Fourth Year/BIA4/ICAs/ICA1/Model and dataset/Model and dataset/test/benign'
malignant_dir = '/Users/A.BC.Perroquet/Library/CloudStorage/OneDrive-InternationalCampus,ZhejiangUniversity/ZJU/Fourth Year/BIA4/ICAs/ICA1/Model and dataset/Model and dataset/test/malignant/'

true_labels_benign = []
predicted_labels_benign = []
true_labels_malignant = []
predicted_labels_malignant = []
predicted_probs = []
tumor_type = ["benign", "malignant"]

# Run over test  (benign)
for filename in os.listdir(benign_dir):
    img_path = os.path.join(benign_dir, filename)

    try:
        # Try opening the image file
        img = torch.unsqueeze(transform(Image.open(img_path).convert("RGB")), dim=0)
    except (Image.UnidentifiedImageError, OSError):
        # Skip the file if it's not a valid image
        print(f"Skipping {img_path} as it is not a valid image file.")
        continue

    # Make a prediction
    with torch.no_grad():
        output = model(img)

        probabilities = F.softmax(output, dim=1)

        pred = torch.argmax(model(img), dim=-1).cpu().numpy()[0]

    true_label = 0

    # Append true and predicted labels to the lists for benign images
    true_labels_benign.append(true_label)
    predicted_labels_benign.append(pred)
    predicted_probs.append(probabilities[0, 1].item())

# print(predicted_probs)
# Run over test  (malignant)
for filename in os.listdir(malignant_dir):
    img_path = os.path.join(malignant_dir, filename)

    try:
        img = torch.unsqueeze(transform(Image.open(img_path).convert("RGB")), dim=0)
    except (Image.UnidentifiedImageError, OSError):
        print(f"Skipping {img_path} as it is not a valid image file.")
        continue

    with torch.no_grad():
        output = model(img)
        # get probabilities for positive
        probabilities = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=-1).cpu().numpy()[0]

    true_label = 1

    true_labels_malignant.append(true_label)
    predicted_labels_malignant.append(pred)
    # get probabilities list for positive
    predicted_probs.append(probabilities[0, 1].item())
# Combine true and predicted labels for both benign and malignant images
true_labels = true_labels_benign + true_labels_malignant
predicted_labels = predicted_labels_benign + predicted_labels_malignant

# print(predicted_probs)

# Calculate the confusion matrix using scikit-learn

confusion_mat = confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp = confusion_mat.ravel()

# Calculate those indicators
print('Accuracy:', accuracy_score(true_labels, predicted_labels))
print('Recall:', recall_score(true_labels, predicted_labels))
print('Precision:', precision_score(true_labels, predicted_labels))
print('Specificity:', tn / (tn + fp))
print('F1 score:', f1_score(true_labels, predicted_labels))
print('AUC:', roc_auc_score(true_labels, predicted_probs))

# draw row
draw_roc(true_labels, predicted_probs)

print("Confusion Matrix:")
print(confusion_mat)
disp = ConfusionMatrixDisplay(confusion_mat)
disp.plot()
plt.show()
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=tumor_type))
