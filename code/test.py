import csv
import time
import warnings
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Cache for loaded datasets
_cached_data = {}

# Function to load and preprocess data
def load_and_preprocess_data(dataset_name):
    if dataset_name in _cached_data:
        return _cached_data[dataset_name]

    if dataset_name == 'mnist':
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
        X = X.astype('float32') / 16.0  # Normalize to [0, 1] range
    elif dataset_name == 'wine':
        wine = datasets.load_wine()
        X, y = wine.data, wine.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
        train_data = next(iter(trainloader))
        test_data = next(iter(testloader))
        X_train, y_train = train_data[0].view(len(trainset), -1).numpy(), train_data[1].numpy()
        X_test, y_test = test_data[0].view(len(testset), -1).numpy(), test_data[1].numpy()
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        X = X.astype('float32') / 2.0 + 0.5  # Normalize to [0, 1] range
    else:
        raise ValueError("Unsupported dataset. Choose from 'mnist', 'wine', 'cifar10'.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _cached_data[dataset_name] = (X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test

# Function to classify data
def classify_data(classifier_name, X_train, y_train, X_test, y_test, time_limit=300):
    start_time = time.time()
    
    if classifier_name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=3)
        try:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        except Exception as e:
            print(f"Training error: {e}")
            return 'NA', 'NA'

    elif classifier_name == 'linear':
        classifier = LogisticRegression(max_iter=1000, solver='liblinear')
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(classifier.fit, X_train, y_train)
                future.result(timeout=60)  # Set a timeout of 60 seconds
            y_pred = classifier.predict(X_test)
        except TimeoutError:
            print("Training timed out")
            return 'NA', 'NA'
        except Exception as e:
            print(f"Training error: {e}")
            return 'NA', 'NA'

    elif classifier_name == 'mahalanobis':
        cov_matrix = np.cov(X_train, rowvar=False)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        X_train_centered = X_train - np.mean(X_train, axis=0)
        X_test_centered = X_test - np.mean(X_train, axis=0)
        try:
            distances = cdist(X_test_centered, X_train_centered, metric='mahalanobis', VI=inv_cov_matrix)
            y_pred = y_train[np.argmin(distances, axis=1)]
        except Exception as e:
            print(f"Distance computation error: {e}")
            return 'NA', 'NA'

    else:
        raise ValueError("Unsupported classifier. Choose from 'knn', 'linear', 'mahalanobis'.")

    return accuracy_score(y_test, y_pred), time.time() - start_time

# Modified pipeline function
def pipeline(params):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(params['dataset_name'])

    # Classify original data (no reduction)
    if params['reduction_method'] == 'none':
        original_accuracy, original_time = classify_data(params['classifier_name'], X_train, y_train, X_test, y_test)
        original_avg_time = 'NA' if original_accuracy == 'NA' else original_time / X_test.shape[0]

        # Write original data classification result to CSV
        write_to_csv(params['csv_file'], [params['dataset_name'], 'none', 'N/A', 'N/A', X_train.shape[1], params['classifier_name'], original_accuracy, 'N/A',f"{original_avg_time:.6f}s per sample"])
        return

    # Apply dimensionality reduction
    start_time = time.time()
    if params['reduction_method'] == 'pca':
        pca = PCA(n_components=params['n_components'])
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
        explained_variance = np.sum(pca.explained_variance_ratio_)
    elif params['reduction_method'] == 'lda':
        n_classes = len(np.unique(y_train))
        lda_n_components = min(n_classes - 1, X_train.shape[1])
        lda = LDA(n_components=lda_n_components)
        X_train_reduced = lda.fit_transform(X_train, y_train)
        X_test_reduced = lda.transform(X_test)
        explained_variance = 'N/A'
    elif params['reduction_method'] == 'pca+lda':
        pca = PCA(n_components=params['n_components'])
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        n_classes = len(np.unique(y_train))
        lda_n_components = min(n_classes - 1, X_train_pca.shape[1])
        lda = LDA(n_components=lda_n_components)
        X_train_reduced = lda.fit_transform(X_train_pca, y_train)
        X_test_reduced = lda.transform(X_test_pca)
        explained_variance = np.sum(pca.explained_variance_ratio_)
    else:
        raise ValueError("Unsupported reduction method. Choose from 'pca', 'lda', 'pca+lda'.")

    reduction_time = time.time() - start_time
    reduction_avg_time = reduction_time / (X_train.shape[0] + X_test.shape[0])

    # Classify reduced data
    reduced_accuracy, classification_time = classify_data(params['classifier_name'], X_train_reduced, y_train, X_test_reduced, y_test)
    classification_avg_time = 'NA' if reduced_accuracy == 'NA' else classification_time / X_test.shape[0]

    # Write reduced data classification result to CSV
    write_to_csv(params['csv_file'], [params['dataset_name'], params['reduction_method'], params['n_components'], explained_variance, X_train_reduced.shape[1], params['classifier_name'], reduced_accuracy, f"{reduction_avg_time:.6f}s per sample", f"{classification_avg_time:.6f}s per sample"])

    print(f"Dataset: {params['dataset_name']}, Reduction: {params['reduction_method']}, Classifier: {params['classifier_name']}, Accuracy: {reduced_accuracy:.2f}")

# Function to write results to CSV
def write_to_csv(csv_file, row_data):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

# Initialize CSV file
def initialize_csv(csv_file='../result/results.csv'):
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Reduction Method', 'Parameters', 'Explained Variance', 'Number of Components', 'Classifier', 'Accuracy', 'Reduction Time per Sample', 'Classification Time per Sample'])

# Run experiments
initialize_csv()

# General settings for datasets, reduction methods, and classifiers
config = {
    'mnist': {'n_components_list': [50, 0.95]},
    'wine': {'n_components_list': [3, 0.95]},
    'cifar10': {'n_components_list': [100, 0.95]}
}

reduction_methods = ['pca', 'lda', 'pca+lda']
classifiers = ['knn', 'linear', 'mahalanobis']

# Generate all experiment combinations (excluding 'none' reduction)
experiments = [
    {
        'dataset_name': dataset,
        'reduction_method': reduction_method,
        'classifier_name': classifier,
        'n_components': n_components,
        'csv_file': '../result/results.csv'
    }
    for dataset, settings in config.items()
    for reduction_method in reduction_methods
    for classifier in classifiers
    for n_components in settings['n_components_list']
]

# Run all experiments (including no reduction classification)
for experiment in experiments:
    pipeline(experiment)

# Also run classification without any reduction
for dataset in config.keys():
    for classifier in classifiers:
        pipeline({
            'dataset_name': dataset,
            'reduction_method': 'none',
            'classifier_name': classifier,
            'n_components': 'N/A',
            'csv_file': '../result/results.csv'
        })