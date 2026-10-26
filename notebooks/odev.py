import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ----------------------------------------------------------------------
# 0. Dosya Yollarını Otomatik Ayarlama (Dizin Sorununu Çözer)
# ----------------------------------------------------------------------
# Kod notebooks/odev.py içinde olsa bile bir üst dizin olan Odev-1'i kök dizin alıyoruz
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

print(f"Proje Kök Dizini: {BASE_DIR}")
print(f"Veri Klasörü: {DATA_DIR}\n")

# ----------------------------------------------------------------------
# 1. Cihaz Seçimi ve Hiperparametreler
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 8
VAL_RATIO = 0.2  # Train verisinin %20'si validation olacak

# ----------------------------------------------------------------------
# 2. Veri Seti Hazırlığı (MNIST + Train/Val/Test Ayrımı)
# ----------------------------------------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# MNIST'i data klasörüne indir/yükle
full_train_dataset = datasets.MNIST(
    root=DATA_DIR, train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transform
)

# Train (%80) ve Validation (%20) ayrımı
val_size = int(len(full_train_dataset) * VAL_RATIO)
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size]
)

# DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(
    f"Veri Ayrımı -> Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}\n"
)


# ----------------------------------------------------------------------
# 3. Temel MLP Model Mimarisi
# ----------------------------------------------------------------------
class MNISTMLP(nn.Module):

  def __init__(self):
    super(MNISTMLP, self).__init__()
    # 2 Hidden Layer: 784 -> 128 -> 64 -> 10
    self.network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

  def forward(self, x):
    return self.network(x)


model = MNISTMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------------------------------------------------
# 4. Model Eğitimi ve Doğrulama (Training & Validation)
# ----------------------------------------------------------------------
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

start_time = time.time()

for epoch in range(EPOCHS):
  # --- Training Phase ---
  model.train()
  running_loss, correct_train, total_train = 0.0, 0, 0
  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * images.size(0)
    _, predicted = torch.max(outputs, 1)
    total_train += labels.size(0)
    correct_train += (predicted == labels).sum().item()

  epoch_train_loss = running_loss / total_train
  epoch_train_acc = 100 * correct_train / total_train
  train_losses.append(epoch_train_loss)
  train_accuracies.append(epoch_train_acc)

  # --- Validation Phase ---
  model.eval()
  val_running_loss, correct_val, total_val = 0.0, 0, 0
  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      val_running_loss += loss.item() * images.size(0)
      _, predicted = torch.max(outputs, 1)
      total_val += labels.size(0)
      correct_val += (predicted == labels).sum().item()

  epoch_val_loss = val_running_loss / total_val
  epoch_val_acc = 100 * correct_val / total_val
  val_losses.append(epoch_val_loss)
  val_accuracies.append(epoch_val_acc)

  print(
      f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
      f"Train Loss: {epoch_train_loss:.4f} - Train Acc: %{epoch_train_acc:.2f} | "
      f"Val Loss: {epoch_val_loss:.4f} - Val Acc: %{epoch_val_acc:.2f}"
  )

total_training_time = time.time() - start_time
print(f"\nToplam Eğitim Süresi: {total_training_time:.2f} saniye")

# ----------------------------------------------------------------------
# 5. Final Test Performansı (Sadece 1 Kere)
# ----------------------------------------------------------------------
model.eval()
correct_test, total_test = 0, 0
with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total_test += labels.size(0)
    correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f"Final Test Accuracy: %{test_accuracy:.2f}\n")

# ----------------------------------------------------------------------
# 6. Loss ve Accuracy Grafiklerinin Çizdirilmesi ve Kaydedilmesi
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Eğitim ve Doğrulama Kaybı (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Eğitim ve Doğrulama Başarımı (Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
metrics_path = os.path.join(BASE_DIR, "metrics_graph.png")
plt.savefig(metrics_path)
plt.show()

# ----------------------------------------------------------------------
# 7. Kendi El Yazısı Görsellerinin Otomatik Tespiti ve Tahmin Edilmesi
# ----------------------------------------------------------------------
print("--- KENDİ EL YAZIN İLE TAHMİN SONUÇLARI ---")

# data/ klasöründeki rakam-*.jpg görsellerini sıralı şekilde bulur
valid_exts = (".jpg", ".jpeg", ".png")
custom_image_paths = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.lower().endswith(valid_exts) and f.lower().startswith("rakam")
])

if len(custom_image_paths) == 0:
  print(
      f"UYARI: {DATA_DIR} klasöründe 'rakam-' ile başlayan görsel bulunamadı!"
  )
else:
  total_custom = len(custom_image_paths)
  plt.figure(figsize=(3 * total_custom, 3))

  for idx, img_path in enumerate(custom_image_paths):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
      print(f"Hata: {img_path} okunamadı!")
      continue

    # MNIST formatına çevirme (Invert -> Resize -> Normalize)
    # Kağıt beyaz (255) yazı siyah (0) ise Invert ile arkaplanı siyah yapıyoruz
    img_inverted = cv2.bitwise_not(img)
    img_resized = cv2.resize(
        img_inverted, (28, 28), interpolation=cv2.INTER_AREA
    )

    img_tensor = transforms.ToTensor()(img_resized)
    img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Model Tahmini
    model.eval()
    with torch.no_grad():
      output = model(img_tensor)
      pred = torch.argmax(output, dim=1).item()

    # Görselleştirme
    file_name = os.path.basename(img_path)
    plt.subplot(1, total_custom, idx + 1)
    plt.imshow(img_resized, cmap="gray")
    plt.title(f"{file_name}\nTahmin: {pred}")
    plt.axis("off")

  plt.tight_layout()
  custom_pred_path = os.path.join(BASE_DIR, "custom_digits_predictions.png")
  plt.savefig(custom_pred_path)
  plt.show()

  print(f"\nİşlem Başarıyla Tamamlandı!")
  print(f"Grafikler '{BASE_DIR}' konumuna kaydedildi.")