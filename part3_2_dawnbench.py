"""
COMP3710 Demo 2 - Part 3.2: DAWNBench Challenge (ResNet-18 on CIFAR-10)
v4: identical to v3 (Cutout augmentation + TTA at final eval), with ONE change:
epochs 24 -> 34. v3's accuracy was still climbing steadily at epoch 24
(91.06% -> 93.18% -> 93.34% -> 93.44%), not plateaued -- the OneCycleLR schedule
finished annealing right as the model was still improving, so this gives it more
room, with the schedule recalculated to match the new epoch count.

Time cost: ~34 * 15s =~ 510s (~8.5 min), noticeably over the lab sheet's 360s
reference point. Worth doing once to see whether the trend actually continues
to/past 94% -- if it does, a well-explained "94%+ in ~510s" is still a strong,
honest result to present even if it doesn't hit the exact 360s figure.

*** REQUIRES A GPU -- RUN ON RANGPUR (comp3710 partition) ***
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import time
import copy
import random

assert torch.cuda.is_available(), "This script requires a GPU -- run it on Rangpur (comp3710 partition)."
device = torch.device('cuda')

# ---------------------------------------------------------------------------
# Cutout augmentation -- randomly zeroes out a square patch of the image
# ---------------------------------------------------------------------------
class Cutout:
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):
        # img is a tensor (C, H, W) at this point, already normalized
        h, w = img.shape[1], img.shape[2]
        cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y1 = max(cy - self.size // 2, 0)
        y2 = min(cy + self.size // 2, h)
        x1 = max(cx - self.size // 2, 0)
        x2 = min(cx + self.size // 2, w)
        img[:, y1:y2, x1:x2] = 0.0
        return img

# ---------------------------------------------------------------------------
# Data: proper 3-way split (unchanged), now with Cutout added to training only
# ---------------------------------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    Cutout(size=8),
])
transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
full_trainset_eval = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_eval)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_eval)

torch.manual_seed(42)
n_val = 5000
n_train = len(full_trainset) - n_val
train_indices, val_indices = random_split(range(len(full_trainset)), [n_train, n_val])

trainset = torch.utils.data.Subset(full_trainset, train_indices.indices)
valset = torch.utils.data.Subset(full_trainset_eval, val_indices.indices)

print(f"Train: {len(trainset)}  Val: {len(valset)}  Test (held out, touch once): {len(testset)}")

trainloader = DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=512, shuffle=False, num_workers=4)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)

model = build_model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

# Reverted to v1's settings -- v2's lower LR / more epochs didn't help
num_epochs = 34
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.7, total_steps=len(trainloader) * num_epochs
)

def evaluate(model, loader, tta=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if tta:
                images_flipped = torch.flip(images, dims=[3])
                with autocast():
                    out1 = F.softmax(model(images), dim=1)
                    out2 = F.softmax(model(images_flipped), dim=1)
                outputs = (out1 + out2) / 2
            else:
                with autocast():
                    outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
epoch_times, val_accuracies = [], []
best_val_acc = 0.0
best_model_state = None
time_to_90_val, time_to_94_val = None, None
total_start = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    val_acc = evaluate(model, valloader)  # plain eval during training, no TTA here
    val_accuracies.append(val_acc)
    elapsed = time.time() - total_start

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

    if val_acc >= 90 and time_to_90_val is None:
        time_to_90_val = elapsed
    if val_acc >= 94 and time_to_94_val is None:
        time_to_94_val = elapsed

    print(f"Epoch {epoch+1}/{num_epochs}: val_acc={val_acc:.2f}%  "
          f"(best so far={best_val_acc:.2f}%)  epoch_time={epoch_time:.1f}s  total_elapsed={elapsed:.1f}s")

total_train_time = time.time() - total_start
print(f"\nBest validation accuracy during training: {best_val_acc:.2f}%")
print(f"Time to reach 90% val accuracy: {time_to_90_val if time_to_90_val else 'not reached'} seconds")
print(f"Time to reach 94% val accuracy: {time_to_94_val if time_to_94_val else 'not reached'} seconds")
print(f"Total training time: {total_train_time:.1f} seconds")

# ---------------------------------------------------------------------------
# FINAL: load best checkpoint, evaluate ONCE on held-out test set -- WITH TTA
# ---------------------------------------------------------------------------
model.load_state_dict(best_model_state)
final_test_acc_plain = evaluate(model, testloader, tta=False)
final_test_acc_tta = evaluate(model, testloader, tta=True)
print(f"\n{'='*55}")
print(f"FINAL HELD-OUT TEST ACCURACY (no TTA):  {final_test_acc_plain:.2f}%")
print(f"FINAL HELD-OUT TEST ACCURACY (with TTA): {final_test_acc_tta:.2f}%   <-- report this")
print(f"{'='*55}")

# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(range(1, num_epochs+1), val_accuracies, 'o-', label='Validation accuracy (no TTA)')
plt.axhline(90, color='orange', linestyle='--', label='90% target')
plt.axhline(94, color='red', linestyle='--', label='94% stretch target')
plt.axhline(final_test_acc_tta, color='green', linestyle=':', label=f'Final test acc w/ TTA ({final_test_acc_tta:.2f}%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('ResNet-18 + Cutout on CIFAR-10 (v4): Validation Accuracy vs Final Test Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('part3_2_accuracy_curve_v5.png', dpi=150)
plt.show()
print("Saved: part3_2_accuracy_curve_v5.png")

plt.figure(figsize=(9, 4))
plt.bar(range(1, num_epochs+1), epoch_times, color='steelblue')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Per-Epoch Training Time (A100, mixed precision, +Cutout) v4')
plt.tight_layout()
plt.savefig('part3_2_epoch_times_v5.png', dpi=150)
plt.show()
print("Saved: part3_2_epoch_times_v5.png")

model.eval()
images, labels = next(iter(testloader))
images, labels = images[:8].to(device), labels[:8].to(device)
with torch.no_grad():
    with autocast():
        outputs = model(images)
    preds = outputs.argmax(dim=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).to(device)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1).to(device)

fig, axes = plt.subplots(1, 8, figsize=(16, 3))
for i in range(8):
    img = (images[i] * std + mean).clamp(0,1).cpu().permute(1,2,0).numpy()
    axes[i].imshow(img)
    axes[i].set_title(f"pred: {classes[preds[i]]}\ntrue: {classes[labels[i]]}", fontsize=9)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('part3_2_inference_demo_v5.png', dpi=150)
plt.show()
print("Saved: part3_2_inference_demo_v5.png")

torch.save(best_model_state, 'best_resnet18_cifar10.pt')
print("Saved: best_resnet18_cifar10.pt (OVERWRITES previous checkpoint -- this is the one")
print("live_demo.py will load; note it does NOT include TTA, which only applies at final eval)")