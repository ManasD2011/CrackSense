"""
Simple Gradio app to test a crack segmentation PyTorch model.

Update the MODEL_PATH and, if needed, the model class/import and preprocessing
to match your project.

Run:
    pip install gradio torch torchvision pillow opencv-python
    python crack_tester.py
"""

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ==========================
# CHANGE THIS
# ==========================
MODEL_PATH = "best_crack_segformer.pt"   # Path to your model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If your checkpoint stores the whole model:
model = torch.load(MODEL_PATH, map_location=device)

# If it stores a checkpoint dict, uncomment and adapt:
# from your_model_file import YourModel
# model = YourModel()
# checkpoint = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict(image):
    original = np.array(image.convert("RGB"))
    h, w = original.shape[:2]

    inp = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(inp)

        # Handle common segmentation outputs
        if isinstance(output, (list, tuple)):
            output = output[0]

        mask = torch.sigmoid(output).squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = original.copy()
    overlay[mask == 1] = [255, 0, 0]  # red crack pixels

    blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

    return Image.fromarray(blended), Image.fromarray((mask * 255).astype(np.uint8))

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Crack Image"),
    outputs=[
        gr.Image(label="Crack Overlay"),
        gr.Image(label="Predicted Mask"),
    ],
    title="Crack Detection Model Tester",
    description="Upload an image to visualize the predicted crack mask."
)

demo.launch()
