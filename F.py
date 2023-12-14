import torch
import tkinter as tk
from tkinter import simpledialog
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

def get_user_input():
    x_user = float(simpledialog.askstring("Input", "Enter x-coordinate:"))
    y_user = float(simpledialog.askstring("Input", "Enter y-coordinate:"))
    return torch.tensor([[x_user, y_user]])

def visualize_decision_regions(model, features, labels, user_input=None):
    plot_decision_regions(X=features, y=labels, clf=model, legend=2)
    if user_input is not None:
        plt.scatter(user_input[:, 0], user_input[:, 1], color='red', marker='x', label='User Input')
    plt.title('Decision Regions of Gaussian Naive Bayes')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x_train = torch.tensor([[-3, 7], [1, 5], [1, 2], [-2, 0], [2, 3], [-4, 0], [-1, 1], [1, 1], [-2, 2], [2, 7], [-4, 1], [-2, 7]])
    y_train = torch.tensor([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

    # Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(x_train.numpy(), y_train.numpy())

    # User input
    user_data = get_user_input()

    # Convert user input to tensor
    user_data_tensor = torch.tensor(user_data)

    # Predict Output
    predicted = model.predict(user_data_tensor.numpy())
    print("User Input: ", user_data_tensor.numpy(), "\nPredicted Class: ", predicted)

    # Visualization of Decision Regions
    visualize_decision_regions(model, x_train.numpy(), y_train.numpy(), user_input=user_data_tensor.numpy())

    # Seaborn Histogram
    sns.histplot(x_train.numpy().flatten(), bins=50, kde=False)
    plt.title('Seaborn Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
