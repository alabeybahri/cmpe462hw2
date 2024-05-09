from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA


def read_images(filename):
    with open(filename, "rb") as file:
        _, num_images, rows, cols = (
            int.from_bytes(file.read(4), "big"),
            int.from_bytes(file.read(4), "big"),
            int.from_bytes(file.read(4), "big"),
            int.from_bytes(file.read(4), "big"),
        )
        images = []
        for _ in range(num_images):
            image = []
            for __ in range(rows * cols):
                pixel = int.from_bytes(file.read(1), "big")
                image.append(pixel)
            images.append(image)
        return images


def read_labels(filename):
    with open(filename, "rb") as file:
        _, num_labels = int.from_bytes(file.read(4), "big"), int.from_bytes(
            file.read(4), "big"
        )
        labels = []
        for _ in range(num_labels):
            label = int.from_bytes(file.read(1), "big")
            labels.append(label)

        return labels


def get_correct_digits(labels, images, correct_digits=[2, 3, 8, 9]):
    result_labels = []
    result_images = []
    for index in range(len(labels)):
        if labels[index] in correct_digits:
            result_labels.append(labels[index])
            result_images.append(images[index])
    return result_images, result_labels


train_images = read_images("train-images.idx3-ubyte")
train_labels = read_labels("train-labels.idx1-ubyte")
test_images = read_images("t10k-images.idx3-ubyte")
test_labels = read_labels("t10k-labels.idx1-ubyte")


correct_train_images, correct_train_labels = get_correct_digits(
    train_labels, train_images
)
correct_test_images, correct_test_labels = get_correct_digits(test_labels, test_images)

X_train = np.array(correct_train_images).reshape(len(correct_train_images), -1)
y_train = np.array(correct_train_labels)
X_test = np.array(correct_test_images).reshape(len(correct_test_images), -1)
y_test = np.array(correct_test_labels)

model = make_pipeline(StandardScaler(), LinearSVC(dual=False, C=1.0, random_state=42))

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy*100}")
print(f"Test Accuracy: {test_accuracy*100}")

pca_components = 50
model_with_pca = make_pipeline(
    StandardScaler(),
    PCA(n_components=pca_components),
    LinearSVC(dual=False, C=1.0, random_state=42),
)

model_with_pca.fit(X_train, y_train)

y_train_pred_pca = model_with_pca.predict(X_train)
y_test_pred_pca = model_with_pca.predict(X_test)

train_accuracy_pca = accuracy_score(y_train, y_train_pred_pca)
test_accuracy_pca = accuracy_score(y_test, y_test_pred_pca)

print(f"Training Accuracy with PCA: {train_accuracy_pca*100}")
print(f"Test Accuracy with PCA: {test_accuracy_pca*100}")

model_with_rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42))

param_grid = {
    "svc__C": [1, 10],
    "svc__gamma": ["scale", 1],
}

grid_search = GridSearchCV(model_with_rbf, param_grid, cv=3, scoring="accuracy")

grid_search.fit(X_train, y_train)

best_model_rbf = grid_search.best_estimator_

y_train_pred_rbf = best_model_rbf.predict(X_train)
y_test_pred_rbf = best_model_rbf.predict(X_test)

train_accuracy_rbf = accuracy_score(y_train, y_train_pred_rbf)
test_accuracy_rbf = accuracy_score(y_test, y_test_pred_rbf)

print(f"Training Accuracy with RBF kernel: {train_accuracy_rbf*100:.2f}%")
print(f"Test Accuracy with RBF kernel: {test_accuracy_rbf*100:.2f}%")
print("Best parameters:", grid_search.best_params_)
