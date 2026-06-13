import torch
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor, class_idx=None):

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()

        output[:, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam.detach().numpy()

        cam = cv2.resize(
            cam,
            (224, 224)
        )

        cam -= cam.min()

        cam /= (cam.max() + 1e-8)

        return cam
    

# Step 3: Preprocess Image
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

image = Image.open("face.jpg").convert("RGB")

input_tensor = transform(image).unsqueeze(0)



# Step 4: Generate Heatmap

# For ResNet:

target_layer = model.layer4[-1]

gradcam = GradCAM(
    model,
    target_layer
)

cam = gradcam.generate(input_tensor)



# Step 5: Overlay Heatmap
original = cv2.imread("face.jpg")
original = cv2.resize(original, (224,224))

heatmap = np.uint8(255 * cam)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)

cv2.imwrite(
    "gradcam_result.jpg",
    overlay
)



