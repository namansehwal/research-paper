# scripts/cnn_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from cnn_data_loader import get_data_loaders
from cnn_model import SimpleCNN

def train_cnn(data_dir, model_save_path, num_epochs=25, batch_size=32, learning_rate=0.001, img_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_data_loaders(data_dir, batch_size, img_size)
    model = SimpleCNN(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved Best Model with Val Acc: {best_val_acc:.4f}")

    print("Training Complete.")

if __name__ == "__main__":
    data_directory = 'dataset/augmented/images'  # Use augmented data for better performance
    model_output_path = 'models/cnn_best.pth'
    os.makedirs('models', exist_ok=True)
    train_cnn(
        data_dir=data_directory,
        model_save_path=model_output_path,
        num_epochs=25,
        batch_size=32,
        learning_rate=0.001,
        img_size=224
    )
