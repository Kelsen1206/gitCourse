"""
COMP3710 Demo 2 - Part 3.2: LIVE DEMO SCRIPT
Run this interactively on Rangpur (comp3710 partition, GPU) in front of your
demonstrator. It does two things, printing progress the whole time so it's
visibly happening live:
  (A) One epoch of real training on the GPU
  (B) Inference using your already-trained best checkpoint

Usage on Rangpur:
    srun --partition=comp3710 --gres=gpu:1 --time=00:15:00 --pty bash
    conda activate torch
    python3 live_demo.py
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import time

assert torch.cuda.is_available(), "No GPU detected -- make sure you requested one with --gres=gpu:1"
device = torch.device('cuda')
print(f"Running on: {torch.cuda.get_device_name(0)}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

def build_model():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)

# ---------------------------------------------------------------------------
# PART A: one epoch of real training, live, with progress printed as it happens
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PART A: Running ONE epoch of training live on the GPU")
print("="*60)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)

demo_model = build_model()
optimizer = torch.optim.SGD(demo_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

demo_model.train()
start = time.time()
for i, (images, labels) in enumerate(trainloader):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    with autocast():
        outputs = demo_model(images)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if i % 10 == 0:
        print(f"  batch {i}/{len(trainloader)}  loss={loss.item():.4f}  elapsed={time.time()-start:.1f}s")

print(f"\nOne epoch complete in {time.time()-start:.1f} seconds -- this demonstrates real")
print("training is actually happening on this Rangpur GPU node, live, right now.")

# ---------------------------------------------------------------------------
# PART B: inference using your already-trained best checkpoint
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PART B: Inference using the saved best checkpoint")
print("="*60)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)

infer_model = build_model()
infer_model.load_state_dict(torch.load('best_resnet18_cifar10.pt', map_location=device))
infer_model.eval()

images, labels = next(iter(testloader))
images, labels = images.to(device), labels.to(device)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

with torch.no_grad():
    with autocast():
        outputs = infer_model(images)
    preds = outputs.argmax(dim=1)

print("\nSample predictions from the trained model (held-out test images):")
correct = 0
for i in range(len(labels)):
    pred_name, true_name = classes[preds[i]], classes[labels[i]]
    match = "CORRECT" if preds[i] == labels[i] else "wrong"
    correct += (preds[i] == labels[i]).item()
    print(f"  Image {i}: predicted={pred_name:8s}  actual={true_name:8s}  [{match}]")

print(f"\n{correct}/{len(labels)} correct on this live sample batch.")
print("\nDemo complete: one training epoch ran live, and the saved best model")
print("performed inference on unseen test images, both on this Rangpur GPU node.")