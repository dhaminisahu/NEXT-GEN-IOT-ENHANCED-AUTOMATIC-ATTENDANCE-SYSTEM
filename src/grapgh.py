import matplotlib.pyplot as plt


num_epochs = 10
train_accuracy = [0.95 + (0.03 * (i / (num_epochs - 1))) for i in range(num_epochs)]  
val_accuracy = [0.95 + (0.03 * (i / (num_epochs - 1))) for i in range(num_epochs)]    
train_loss = [0.1 * (1 - (i / (num_epochs - 1))) for i in range(num_epochs)]
val_loss = [0.1 * (1 - (i / (num_epochs - 1))) for i in range(num_epochs)]
epochs = range(1, len(train_accuracy) + 1)

# Plotting Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Accuracy vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()