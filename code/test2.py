import os
import time
import csv
import logging
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

path = '/media/user/volume2/students/s124md209_01/WangShengyuan/6222/Assignment1/'
# Constants
RESULTS_CSV_PATH = os.path.join(path, 'result/results1.csv')
DATA_DIR = os.path.join(path, 'data')

# Cache for loaded datasets
_cached_data = {}

def main():
    """
    Main function to run all experiments.
    """
    initialize_csv(RESULTS_CSV_PATH)
    config = {
        'mnist': {'n_components_list': [20, 30, 40, 50, 0.95]},
        'wine': {'n_components_list': [2, 3, 5, 0.95]},
        'cifar10': {'n_components_list': [50, 100, 200, 0.95]}
    }

    classifiers = ['knn', 'linear', 'mahalanobis']

    # Generate all experiments
    experiments = []
    for dataset_name, settings in config.items():
        experiments.extend(generate_experiments(dataset_name, settings['n_components_list'], classifiers))

    # Include CNN feature extraction and classification for CIFAR-10
    if 'cifar10' in config:
        cnn_experiments = generate_cnn_experiments('cifar10', classifiers)
        experiments.extend(cnn_experiments)

    # Run all experiments
    for experiment in experiments:
        logging.info(f"Starting experiment: {experiment}")
        try:
            pipeline(experiment)
        except Exception as e:
            logging.error(f"Error in experiment {experiment}: {e}")
        logging.info(f"Finished experiment: {experiment}")

def generate_experiments(dataset_name, n_components_list, classifiers):
    """
    Generate experiment configurations for a given dataset.
    """
    experiments = []
    # Define reduction methods and their corresponding n_components
    reduction_methods = {
        'none': [None],
        'lda': [None],
        'pca': n_components_list,
        'pca+lda': n_components_list
    }
    for reduction_method, n_components_values in reduction_methods.items():
        for n_components in n_components_values:
            for classifier_name in classifiers:
                experiments.append({
                    'dataset_name': dataset_name,
                    'reduction_method': reduction_method,
                    'classifier_name': classifier_name,
                    'n_components': n_components
                })
    return experiments

def generate_cnn_experiments(dataset_name, classifiers):
    """
    Generate experiments using CNN features for traditional classifiers.
    """
    experiments = []
    for classifier_name in classifiers:
        experiments.append({
            'dataset_name': dataset_name,
            'reduction_method': 'cnn_features',
            'classifier_name': classifier_name,
            'n_components': None
        })
    return experiments

def initialize_csv(csv_file):
    """
    Initialize the CSV file with headers.
    """
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Dataset',
            'Reduction Method',
            'Parameters',
            'Explained Variance',
            'Number of Components',
            'Classifier',
            'Accuracy',
            'Reduction Time per Sample',
            'Classification Time per Sample'
        ])
    logging.info(f"Initialized CSV file at {csv_file}")

def pipeline(params):
    """
    Pipeline to run the experiment with the given parameters.
    """
    dataset_name = params['dataset_name']
    reduction_method = params['reduction_method']
    classifier_name = params['classifier_name']
    n_components = params['n_components']

    logging.info(f"Pipeline started for dataset: {dataset_name}, reduction: {reduction_method}, classifier: {classifier_name}, n_components: {n_components}")

    # Handle CNN feature extraction
    if reduction_method == 'cnn_features':
        # Load data and extract CNN features
        X_train, X_test, y_train, y_test = extract_cnn_features(dataset_name)
        # Classify using traditional classifiers
        accuracy, classification_time = classify_data(
            classifier_name, X_train, y_train, X_test, y_test, dataset_name, reduction_method
        )
        classification_avg_time = 'NA' if accuracy == 'NA' else f"{classification_time / X_test.shape[0]:.6f}s"
        # Write results to CSV
        write_to_csv(RESULTS_CSV_PATH, [
            dataset_name,
            reduction_method,
            'N/A',
            'N/A',
            X_train.shape[1],
            classifier_name,
            accuracy,
            'N/A',
            classification_avg_time
        ])
        logging.info(f"Dataset: {dataset_name}, Reduction: {reduction_method}, "
                     f"Classifier: {classifier_name}, Accuracy: {accuracy}, "
                     f"Classification time per sample: {classification_avg_time}")
        return

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_name)

    if reduction_method == 'none':
        # Classify without dimensionality reduction
        accuracy, total_time = classify_data(
            classifier_name, X_train, y_train, X_test, y_test, dataset_name, reduction_method
        )
        avg_time_per_sample = 'NA' if accuracy == 'NA' else f"{total_time / X_test.shape[0]:.6f}s"
        write_to_csv(RESULTS_CSV_PATH, [
            dataset_name,
            'none',
            'N/A',
            'N/A',
            X_train.shape[1],
            classifier_name,
            accuracy,
            'N/A',
            avg_time_per_sample
        ])
        logging.info(f"Dataset: {dataset_name}, Classifier: {classifier_name}, "
                     f"Accuracy: {accuracy}, Time per sample: {avg_time_per_sample}")
        return

    # Apply dimensionality reduction
    X_train_reduced, X_test_reduced, explained_variance, reduction_avg_time = apply_dimensionality_reduction(
        reduction_method, X_train, X_test, y_train, n_components
    )

    # After dimensionality reduction
    scaler = StandardScaler()
    X_train_reduced = scaler.fit_transform(X_train_reduced)
    X_test_reduced = scaler.transform(X_test_reduced)

    # Classify reduced data
    accuracy, classification_time = classify_data(
        classifier_name, X_train_reduced, y_train, X_test_reduced, y_test, dataset_name, reduction_method
    )
    classification_avg_time = 'NA' if accuracy == 'NA' else f"{classification_time / X_test_reduced.shape[0]:.6f}s"

    # Write results to CSV
    write_to_csv(RESULTS_CSV_PATH, [
        dataset_name,
        reduction_method,
        n_components,
        explained_variance,
        X_train_reduced.shape[1],
        classifier_name,
        accuracy,
        f"{reduction_avg_time:.6f}s",
        classification_avg_time
    ])
    logging.info(f"Dataset: {dataset_name}, Reduction: {reduction_method}, "
                 f"Classifier: {classifier_name}, Accuracy: {accuracy}, "
                 f"Reduction time per sample: {reduction_avg_time:.6f}s, "
                 f"Classification time per sample: {classification_avg_time}")

def load_and_preprocess_data(dataset_name):
    """
    Load and preprocess data for the given dataset.
    """
    if dataset_name in _cached_data:
        return _cached_data[dataset_name]

    if dataset_name == 'mnist':
        X, y = load_mnist()
    elif dataset_name == 'wine':
        X, y = load_wine()
    elif dataset_name == 'cifar10':
        X, y = load_cifar10()
    else:
        raise ValueError("Unsupported dataset. Choose from 'mnist', 'wine', 'cifar10'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    _cached_data[dataset_name] = (X_train, X_test, y_train, y_test)
    logging.info(f"Loaded and preprocessed dataset: {dataset_name}")
    return X_train, X_test, y_train, y_test

def load_mnist():
    """
    Load and preprocess the MNIST dataset.
    """
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data.astype('float32') / 16.0  # Normalize to [0, 1]
    y = digits.target
    return X, y

def load_wine():
    """
    Load and preprocess the Wine dataset.
    """
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def load_cifar10():
    """
    Load and preprocess the CIFAR-10 dataset for traditional classifiers.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(dataloader))
    X = X.view(len(dataset), -1).numpy()
    y = y.numpy()
    X = X.astype('float32') / 2.0 + 0.5  # Normalize to [0, 1]
    return X, y

def extract_cnn_features(dataset_name):
    """
    Extract features from CNN for CIFAR-10 dataset.
    """
    if dataset_name != 'cifar10':
        raise ValueError("CNN feature extraction is only implemented for CIFAR-10.")

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    # Data loaders
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # Define CNN model (using a pretrained model)
    model = torchvision.models.resnet18(pretrained=True)
    model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Remove the final classification layer
    features = torch.nn.Sequential(*list(model.children())[:-1])

    # Function to extract features
    def get_features(loader):
        X_features = []
        y_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = features(inputs)
                outputs = outputs.view(outputs.size(0), -1)
                X_features.append(outputs.cpu().numpy())
                y_labels.append(labels.numpy())
        X_features = np.concatenate(X_features, axis=0)
        y_labels = np.concatenate(y_labels, axis=0)
        return X_features, y_labels

    # Extract features
    X_train, y_train = get_features(trainloader)
    X_test, y_test = get_features(testloader)

    return X_train, X_test, y_train, y_test

def apply_dimensionality_reduction(method, X_train, X_test, y_train, n_components):
    """
    Apply the specified dimensionality reduction method to the data.
    """
    start_time = time.time()
    explained_variance = 'N/A'
    if method == 'pca':
        X_train_reduced, X_test_reduced, explained_variance = apply_pca(
            X_train, X_test, n_components
        )
    elif method == 'lda':
        X_train_reduced, X_test_reduced = apply_lda(X_train, X_test, y_train)
    elif method == 'pca+lda':
        X_train_pca, X_test_pca, explained_variance = apply_pca(X_train, X_test, n_components)
        X_train_reduced, X_test_reduced = apply_lda(X_train_pca, X_test_pca, y_train)
    else:
        raise ValueError("Unsupported reduction method. Choose from 'pca', 'lda', 'pca+lda'.")
    reduction_time = time.time() - start_time
    reduction_avg_time = reduction_time / (X_train.shape[0] + X_test.shape[0])
    logging.info(f"Applied {method} reduction in {reduction_time:.2f}s")
    return X_train_reduced, X_test_reduced, explained_variance, reduction_avg_time

def apply_pca(X_train, X_test, n_components):
    """
    Apply PCA to the data.
    """
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logging.info(f"PCA explained variance: {explained_variance:.2f}")
    return X_train_reduced, X_test_reduced, explained_variance

def apply_lda(X_train, X_test, y_train):
    """
    Apply LDA to the data.
    """
    n_classes = len(np.unique(y_train))
    lda_n_components = min(n_classes - 1, X_train.shape[1])
    lda = LDA(n_components=lda_n_components)
    X_train_reduced = lda.fit_transform(X_train, y_train)
    X_test_reduced = lda.transform(X_test)
    logging.info(f"LDA reduced data to {lda_n_components} components")
    return X_train_reduced, X_test_reduced

def classify_data(classifier_name, X_train, y_train, X_test, y_test, dataset_name, reduction_method):
    """
    Train and evaluate the specified classifier.
    """
    start_time = time.time()
    try:
        if classifier_name == 'knn':
            logging.info(f"Training KNN classifier on dataset: {dataset_name} with reduction: {reduction_method}")
            classifier = KNeighborsClassifier(n_neighbors=3)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        elif classifier_name == 'linear':
            logging.info(f"Training LogisticRegression on dataset: {dataset_name} with reduction: {reduction_method}")
            # Adjust parameters for high-dimensional data
            if (dataset_name == 'cifar10') and (reduction_method == 'none' or X_train.shape[1] > 500):
                logging.info("Using 'saga' solver for high-dimensional data")
                # Limit training data size to speed up
                max_samples = 5000
                if X_train.shape[0] > max_samples:
                    logging.info(f"Reducing training samples to {max_samples}")
                    X_train = X_train[:max_samples]
                    y_train = y_train[:max_samples]
                classifier = LogisticRegression(
                    max_iter=100, 
                    solver='saga', 
                    tol=1e-2, 
                    multi_class='multinomial',
                    n_jobs=-1
                )
            else:
                logging.info("Using default 'lbfgs' solver")
                classifier = LogisticRegression(
                    max_iter=200, 
                    solver='lbfgs', 
                    multi_class='auto',
                    n_jobs=-1
                )
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        elif classifier_name == 'mahalanobis':
            logging.info(f"Training Mahalanobis classifier on dataset: {dataset_name} with reduction: {reduction_method}")
            # Limit sample size and dimensions for Mahalanobis classifier on CIFAR-10
            if dataset_name == 'cifar10':
                if X_train.shape[1] > 50:
                    logging.warning("Data dimension too high for Mahalanobis classifier on CIFAR-10. Skipping.")
                    return 'NA', 'NA'
                else:
                    max_samples = 5000  # Adjust based on your computational resources
                    if X_train.shape[0] > max_samples:
                        logging.warning(f"Reducing training samples to {max_samples} for Mahalanobis classifier.")
                        X_train = X_train[:max_samples]
                        y_train = y_train[:max_samples]
            y_pred = mahalanobis_classifier(X_train, y_train, X_test)
        else:
            raise ValueError("Unsupported classifier. Choose from 'knn', 'linear', 'mahalanobis'.")
        accuracy = accuracy_score(y_test, y_pred)
        total_time = time.time() - start_time
        logging.info(f"Classifier {classifier_name} achieved accuracy {accuracy:.4f} in {total_time:.2f}s")
        return accuracy, total_time
    except Exception as e:
        logging.error(f"Error in classification with {classifier_name}: {e}")
        return 'NA', 'NA'

def mahalanobis_classifier(X_train, y_train, X_test):
    """
    Classify using the Mahalanobis distance.
    """
    cov_matrix = np.cov(X_train, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    X_train_centered = X_train - np.mean(X_train, axis=0)
    X_test_centered = X_test - np.mean(X_train, axis=0)
    distances = cdist(
        X_test_centered, X_train_centered, metric='mahalanobis', VI=inv_cov_matrix
    )
    y_pred = y_train[np.argmin(distances, axis=1)]
    return y_pred

def write_to_csv(csv_file, row_data):
    """
    Write a row of data to the CSV file.
    """
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)
    logging.debug(f"Wrote data to CSV: {row_data}")

if __name__ == '__main__':
    main()
