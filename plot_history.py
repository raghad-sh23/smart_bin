import json
import matplotlib.pyplot as plt

# 1. Load history
with open("history.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["accuracy"]) + 1)

# 2. Accuracy curve
plt.figure()
plt.plot(epochs, history["accuracy"], label="Training accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy (MobileNetV2)")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png", bbox_inches="tight")
plt.close()

# 3. Loss curve
plt.figure()
plt.plot(epochs, history["loss"], label="Training loss")
plt.plot(epochs, history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (MobileNetV2)")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", bbox_inches="tight")
plt.close()

print("Saved accuracy_curve.png and loss_curve.png")
