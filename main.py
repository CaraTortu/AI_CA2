import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def show_3d_evaluate(model, nn: bool, dataset: pd.DataFrame, dataset_pca) -> None:
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(111, projection="3d")
    xyz = []

    for i in range(-5, 15):
        for j in range(-10, 10):
            for k in range(-5, 10):
                xyz.append([i, j, k])
    if nn:
        predicted = [i[0]-i[1] for i in model.predict(xyz, verbose=0)]
    else:
        predicted = model.predict(xyz)

    axis.scatter([i[0] for i in xyz], [i[1] for i in xyz], [i[2] for i in xyz], c=predicted, alpha=0.1)
    axis.scatter(dataset_pca[:, 0], dataset_pca[:, 1], dataset_pca[:, 2], c=dataset.get(["diagnosis"]), cmap="coolwarm")
    plt.show()



# Constants
DATASET_SOURCE = "dataset/breast-cancer.csv"
MALIGNANT = 1
BENING = 0

# Read the dataset
dataset = pd.read_csv(DATASET_SOURCE)

# Remove the ID column since pandas already has row numbers
dataset = dataset.drop("id", axis=1)

# Set M to MALIGNANT value and Bening to BENING Value
dataset = dataset.replace("M", MALIGNANT)
dataset = dataset.replace("B", BENING)

# Show heatmap of correlation between the diagnosis and the other features
# This will show us how much each feature affects the diagnosis. 
# It allows us to remove the fields whose impact in the diagnosis is minimal
# letting us reduce the complexity of our features.
matrix = pd.DataFrame(dataset.corr()["diagnosis"], columns=["diagnosis"])
matrix = matrix.drop(["diagnosis"], axis=0)
matrix = matrix.sort_values(by=["diagnosis"], ascending=False)
colormap = sns.color_palette("light:#000", 128, as_cmap=True)
sns.heatmap(matrix, cmap=colormap, annot=True,
            xticklabels=True, yticklabels=True)
plt.show()
plt.close()


# Remove the irrelevant fields based on the correlation map from before
dataset = dataset.drop(["fractal_dimension_mean", "fractal_dimension_se",
                       "texture_se", "smoothness_se", "symmetry_se"], axis=1)

# Get target and labels
labels = list(dataset.columns)

# We want to scale our data in order to achieve a mean of 0 and a variance of 1.
# This can highly improve the accuracy of our algorithm like pictured on figure 1.
# Figure 1: https://scikit-learn.org/stable/_images/sphx_glr_plot_scaling_importance_001.png
dataset_sc = StandardScaler().fit_transform(dataset.loc[:, labels].values)

# Currently our data has a high dimensionality and, in order to visualise it and
# reduce compute times, we can to lower this dimensionality to 3 dimensions at most
# so we can visualise it.
pca = PCA(n_components=3)
dataset_pca = pca.fit_transform(dataset_sc)

# Show the 3 eigenplanes in an interactive view.
fig = plt.figure(figsize=(10, 10))
axis = fig.add_subplot(111, projection="3d")
axis.scatter(dataset_pca[:, 0], dataset_pca[:, 1], dataset_pca[:, 2], c=dataset.get(["diagnosis"]), cmap="coolwarm")
axis.set_xlabel("Principal_component_1", fontsize=11)
axis.set_ylabel("Principal_component_2", fontsize=11)
axis.set_zlabel("Principal_component_3", fontsize=11)
plt.show()

print("\nSTATISTICS:\n")

# Observe the variance values and the accuracy that we have over the original amount of features 
print("We kept {:.2f}% of the data by reducing the features from {} to {}".format(
    sum(pca.explained_variance_ratio_), len(labels), len(pca.explained_variance_ratio_)))

# Get our X and Y
y = dataset.loc[:, ["diagnosis"]]
x = dataset_pca

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Neural network
# Declare the model with 2 output neurons and 1 hidden layer with 32 neurons.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Add the loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=0)

# Evaluate accuracy with test data
nn_score = model.evaluate(x_test, y_test, verbose=0)

# show model
show_3d_evaluate(model, True, dataset, dataset_pca)
sns.heatmap(confusion_matrix(y_test, model.predict(x_test).argmax(axis=1)), annot=True, xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.title("NN confusion matrix")
plt.show()

# SVM
svm = SVC()
svm.fit(x_train, y_train)

# Calculate accuracy_score
svm_score = accuracy_score(y_test, svm.predict(x_test))

# Show model 
show_3d_evaluate(svm, False, dataset, dataset_pca)
sns.heatmap(confusion_matrix(y_test, svm.predict(x_test)), annot=True, xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.title("SVM confusion matrix")
plt.show()


# KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

# Calculate accuracy score
neigh_score = accuracy_score(y_test, neigh.predict(x_test))

# Show model
show_3d_evaluate(neigh, False, dataset, dataset_pca)
sns.heatmap(confusion_matrix(y_test, neigh.predict(x_test)), annot=True, xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.title("KNN confusion matrix")
plt.show()

# Print accuracies
print(f"Neural network accuracy: {nn_score[1]:.2f}%")
print(f"SVM accuracy: {svm_score:.2f}%")
print(f"KNN accuracy: {neigh_score:.2f}%")


