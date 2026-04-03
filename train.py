import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Import your custom modules
import config
from dataset import get_dataloader
from model import CIFAR10CNN
from engine import train_one_epoch


def main():
    # 1. Hardware Configuration
    # Automatically use the GPU if you have one, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    # 2. Load the Data Pipeline
    print("Loading CIFAR-10 dataset...")
    # Make sure 'train' folder and 'trainLabels.csv' are in the same directory
    train_loader = get_dataloader("train", "trainLabels.csv", num_workers=0)

    # 3. Initialize the Model
    print("Initializing CNN architecture...")
    model = CIFAR10CNN(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = StepLR(
        optimizer,
        step_size=config.STEP_LR_STEP_SIZE,
        gamma=config.STEP_LR_GAMMA,
    )

    # 4. The Master Training Loop
    epochs = config.EPOCHS
    print(
        f"Beginning training for {epochs} epochs "
        f"(lr={config.LEARNING_RATE}, StepLR every {config.STEP_LR_STEP_SIZE} epochs, gamma={config.STEP_LR_GAMMA})...\n"
    )

    for epoch in range(epochs):
        avg_loss, avg_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f} | "
            f"Train Accuracy: {(avg_acc * 100):.2f}% | LR: {lr:.6f}"
        )

    print("\nTraining complete! Saving model weights...")
    torch.save(model.state_dict(), "cifar10_model.pth")
    print("Model successfully saved to 'cifar10_model.pth'.")


if __name__ == "__main__":
    main()
