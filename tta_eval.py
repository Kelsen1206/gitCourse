"""
COMP3710 Demo 2 - Part 3.2: Test-Time Augmentation (TTA)
Squeezes extra accuracy out of your ALREADY-TRAINED checkpoint, with no retraining.
Averages predictions over the original test image and its horizontal flip.

This is a legitimate evaluation technique -- it only touches the model at
inference time, never sees test labels during any training, and doesn't change
what "held-out test accuracy" means. Takes under a minute to run.

*** RUN ON RANGPUR (GPU) ***
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast

assert torch.cuda.is_available(), "Run this on Rangpur (comp3710 partition)."
device = torch.device('cuda')

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_eval)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

def build_model():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)

model = build_model()
model.load_state_dict(torch.load('best_resnet18_cifar10.pt', map_location=device))
model.eval()

# ---------------------------------------------------------------------------
# Baseline: accuracy WITHOUT test-time augmentation (should match your training run's number)
# ---------------------------------------------------------------------------
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
baseline_acc = 100 * correct / total
print(f"Baseline test accuracy (no TTA): {baseline_acc:.2f}%")

# ---------------------------------------------------------------------------
# TTA: average softmax probabilities over original + horizontally flipped image
# ---------------------------------------------------------------------------
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        images_flipped = torch.flip(images, dims=[3])  # flip along width

        with autocast():
            out1 = F.softmax(model(images), dim=1)
            out2 = F.softmax(model(images_flipped), dim=1)
        avg_out = (out1 + out2) / 2
        preds = avg_out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
tta_acc = 100 * correct / total
print(f"Test-time augmentation accuracy (original + flip averaged): {tta_acc:.2f}%")
print(f"Improvement from TTA: +{tta_acc - baseline_acc:.2f} percentage points")
