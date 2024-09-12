import timm
import torch
import time

# List models
models = timm.list_models()

# Search for mobilenet variants
mobilenet_models = [model for model in models if 'mobilenetv4' in model.lower()]
print(mobilenet_models)

# Remove the classification head
model = timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params/1e6}M")

image = torch.randn(1, 3, 224, 224)
image = image.to(device)

while True:
    start = time.time()
    with torch.no_grad():
        output = model(image)
    print(f"Time: {time.time() - start}")
    print(f"Frequency: {1/(time.time() - start)}")
    input("Press Enter to continue...")
