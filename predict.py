import torch
from torchvision import transforms
from PIL import Image
from model import build_model
import sys

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

model = build_model()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

image = Image.open(sys.argv[1]).convert("RGB")
tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(tensor).sigmoid().item()

label = "dog" if output > 0.5 else "cat"
confidence = output if output > 0.5 else 1 - output
print(f"{label} ({confidence:.1%} confident)")