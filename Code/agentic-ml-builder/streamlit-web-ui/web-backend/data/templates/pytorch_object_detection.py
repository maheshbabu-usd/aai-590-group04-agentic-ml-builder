
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

# 1. Load Pre-trained Model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. Load Image
# image = Image.open('test.jpg')
# Creating synthetic image for demo
image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
image_tensor = F.to_tensor(image).unsqueeze(0)

# 3. Inference
with torch.no_grad():
    predictions = model(image_tensor)

# 4. Process Results
# boxes, labels, scores
print("Detected boxes:", predictions[0]['boxes'])
print("Scores:", predictions[0]['scores'])

# 5. Visualize (simplified)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1)
ax.imshow(image)

for box in predictions[0]['boxes']:
    rect = patches.Rectangle(
        (box[0], box[1]), 
        box[2]-box[0], box[3]-box[1], 
        linewidth=1, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
plt.show()
