import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def get_confusion_matrix(model, test_loader, class_names):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    # save history (future proof)
    np.save("y_true.npy", y_true)
    np.save("y_pred.npy", y_pred)

    # plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return cm
