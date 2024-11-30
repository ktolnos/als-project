import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms
import numpy as np
# from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import os
import librosa
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, f1_score, confusion_matrix


# Move model to device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)

    # Truncate the frequency at 5 kHz and calculate Mel spectogram
    ms = librosa.feature.melspectrogram(y=y, sr=sr, fmax=5000)

    # Convert spectogram to decibel scale
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close()
    
def load_images_from_path(path, label):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    if os.path.isfile(path):  # Ensure it's a file
        img = Image.open(path).convert("RGB")  # Ensure 3 channels
        image = transform(img)  # Apply transformations
        # Ensure that the label is returned properly
        return image, label
    return None, None  # Return None if path is not a valid file

def show_images(images, max_cols=8):
    """
    Display a grid of images from a PyTorch tensor.
    Args:
        images: Tensor of shape (N, C, H, W), where N is the number of images.
        max_cols: Maximum number of columns in the grid.
    """
    n_images = len(images)
    n_cols = min(max_cols, n_images)  # Limit the number of columns
    n_rows = (n_images + n_cols - 1) // n_cols  # Calculate rows needed
    print(f"Number of images: {n_images}, number of columns and rows: {n_cols, n_rows}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), subplot_kw={'xticks': [], 'yticks': []})
    axes = axes.flatten()  # Flatten to iterate over
    
    for i, ax in enumerate(axes):
        if i < n_images:
            ax.imshow(images[i].permute(1, 2, 0).numpy())  # Convert tensor to NumPy
        else:
            ax.axis('off')  # Hide extra axes
    
    plt.tight_layout()
    plt.show(block=False)

def save_spectograms(paths, folder_to_create="spectograms"):

    new_file_paths = []

    for file in paths:
        # print("Original file:",file)

        relative_path = os.path.dirname(file)

        output_path =os.path.join(relative_path, folder_to_create)

        if not os.path.exists(output_path):
                os.makedirs(output_path)

        png_filepath = os.path.join(output_path,os.path.basename(file)).replace(".wav", ".png")
        new_file_paths.append(png_filepath)
        

        if os.path.isfile(png_filepath):
             print("New file exists.")
        elif not os.path.isfile(png_filepath):
            create_spectrogram(file, png_filepath)
            print("New file", png_filepath)
        else:
             print("An error occured")

    return new_file_paths

def get_images_labels(image_paths, labels):

    x = []  # List to store images
    y = []  # List to store labels

    for path, label in zip(image_paths, labels):
        # print(f"Label: {label}, Path: {path}")

        # Load the image and label
        image, label = load_images_from_path(path, label)
        
        if image is not None:  # Check if the image was loaded successfully
            x.append(image)
            y.append(label)

    # Convert x and y to tensors if needed for training
    x = torch.stack(x)  # Stack images into a tensor
    y = torch.tensor(y)  # Convert labels to a tensor

    # Optional: Check shapes
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    return x, y

def augment_and_save_spectograms(image_paths, labels, augmented_folder="augmented_spectograms"):
    " Augment images by creating horizontal flips and save their spectograms"

    x = []
    y = []

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    new_file_paths_aug = []

    for file in image_paths:
        print("Original file:",file)

        relative_path = os.path.dirname(file)

        output_path =os.path.join(relative_path, augmented_folder)

        if not os.path.exists(output_path):
                os.makedirs(output_path)

        png_filepath = os.path.join(output_path,os.path.basename(file)).replace(".wav", ".png")
        new_file_paths.append(png_filepath)
        

        if os.path.isfile(png_filepath):
             print("New file exists.")
        elif not os.path.isfile(png_filepath):
            create_spectrogram(file, png_filepath)
            print("New file", png_filepath)
        else:
             print("An error occured")

    return new_file_paths

# Transform values: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
# Resnet specs : https://pytorch.org/hub/pytorch_vision_resnet/

class ModelTrainer:
    def __init__(self, model, x, y, dataset, phoneme, num_epochs=10, batch_size=32, lr=0.0001, model_type="resnet"):

        print("Model specified:", model_type)

        self.log_file = open(f'{model_type}_spectogram_training_log_{dataset[0]}_{phoneme[0].lower()}.txt', 'w')

        self.val_roc_aucs = []
        self.confusion_matrices = []

         # Initialize lists to store accuracy values
        self.train_accuracies = []
        self.val_accuracies = []
        self.num_epochs = num_epochs
        self.epochs_loss = []

        # Define class labels for visualization
        self.id_to_label = {i: f"Class {i}" for i in range(len(np.unique(y)))}

        # Preprocessing function for MobileNetV2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        x_scaled = x / 255.0
        x_norm = torch.stack([self.transform(img) for img in x_scaled])  
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Stratified sampling for train-test split
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(stratified_split.split(x_norm, y))

        # Remove augmented data from test dataset
        test_idx = list(set(test_idx) - set(augmented_idxs))

        # Split data into train and test
        x_train, y_train = x_norm[train_idx], y_tensor[train_idx]
        x_test, y_test = x_norm[test_idx], y_tensor[test_idx]

        self.train_dataset = TensorDataset(x_train, y_train)
        self.test_dataset = TensorDataset(x_test, y_test)

        # Stratified DataLoader for training set
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.stratified_sampler(y_train, batch_size))
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # MobileNetV2 setup
        self.model = model
        self.model.to(device)

        if model_type == "resnet":
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, len(np.unique(y)))  
            ).to(device)
        elif model_type == "mobilenet":
            print("Mobilenet specs used.")
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.last_channel, len(np.unique(y)))  
            ).to(device)
        else:
            print("The model was not specified")

        # Set up loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def visualise_dataloader(self, dl, id_to_label=None, with_outputs=True):
        total_num_images = len(dl.dataset)
        class_batch_counts = []  # Store counts of all classes per batch
        idxs_seen = []

        for i, batch in enumerate(dl):
            inputs, labels = batch  # Unpack inputs and labels from the batch
            idxs = list(range(i * dl.batch_size, i * dl.batch_size + len(labels)))
            idxs_seen.extend(idxs)

            # Count occurrences of each class in the batch
            class_ids, class_counts = labels.unique(return_counts=True)
            counts = dict(zip(class_ids.tolist(), class_counts.tolist()))

            # Add counts for all possible classes (fill missing ones with 0)
            all_class_counts = [counts.get(cls, 0) for cls in range(len(id_to_label))]
            class_batch_counts.append(all_class_counts)

        if with_outputs:
            class_batch_counts = np.array(class_batch_counts).T  # Transpose for easier plotting
            fig, ax = plt.subplots(figsize=(15, 8))

            width = 0.35
            ind = np.arange(len(class_batch_counts[0]))  # Number of batches

            # Plot bars for each class
            for cls, counts in enumerate(class_batch_counts):
                ax.bar(
                    ind + cls * width,  # Shift bars for each class
                    counts,
                    width,
                    label=(id_to_label[cls] if id_to_label is not None else f"Class {cls}"),
                )

            ax.set_xticks(ind + width * (len(class_batch_counts) - 1) / 2)
            ax.set_xticklabels(ind + 1)
            ax.set_xlabel("Batch Index", fontsize=12)
            ax.set_ylabel("No. of Images in Batch", fontsize=12)
            ax.legend()
            plt.title("Class Distribution Across Batches", fontsize=15)
            plt.show(block=False)

            num_images_seen = len(set(idxs_seen))
            print(f"Num. unique images seen: {num_images_seen}/{total_num_images}")

        return class_batch_counts, idxs_seen

    def stratified_sampler(self, labels, batch_size):
        # Create a stratified sampler
        class_counts = np.bincount(labels.numpy())
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels.numpy()]
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return sampler

    def train(self):

        
        all_labels = []
        all_predictions = []

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.epochs_loss.append(epoch_loss)
            epoch_acc = accuracy_score(all_labels, all_predictions)


            self.train_accuracies.append(epoch_acc)  # Store training accuracy
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

            # Validation loop
            self.model.eval() 

            all_labels = []
            all_predictions = []
            all_probabilities = []

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    probabilities = torch.softmax(outputs, dim=1)[:, 1] 
        

                    # Collect the predictions and true labels
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

            # Calculate performance metrics
            val_acc = accuracy_score(all_labels, all_predictions)
            roc_auc = self.get_roc_auc(all_labels, all_probabilities)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            cm = self.get_confusion_matrix(all_labels, all_predictions)

            self.val_accuracies.append(val_acc)  # Store validation accuracy

            report = classification_report(
            all_labels, all_predictions, target_names=list(self.id_to_label.values())
            )

            # Append metrics to the log file
            self.log_file.write(f'Epoch {epoch + 1}/{self.num_epochs}\n')
            self.log_file.write(f'Training Loss: {epoch_loss:.4f}\n')
            self.log_file.write(f'Validation Accuracy: {val_acc:.4f}\n')
            self.log_file.write(f'ROC AUC: {roc_auc:.4f}\n')
            self.log_file.write(f'F1 Score: {f1:.4f}\n')
            self.log_file.write(f'Confusion matrix: \n')
            self.log_file.write(np.array2string(cm) + '\n')
            self.log_file.write('Classification Report:\n')
            self.log_file.write(report + '\n')
            self.log_file.write('-' * 50 + '\n')

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f},Validation Accuracy: {val_acc:.4f}, ROC AUC: {roc_auc:.4f}')
            print(report)

        # Close the log file
        self.log_file.write(f"Epochs loss: {self.epochs_loss}")
        self.log_file.write('\n'+ '-' * 50 + '\n')
        self.log_file.write(f"Train Accuracy: {self.train_accuracies}")
        self.log_file.write('\n' + '-' * 50 + '\n')
        self.log_file.write(f"Validation Accuracy: {self.val_accuracies}")
        
        self.log_file.close()
        # self.plot_auc_roc(all_labels, all_probabilities)
        # self.plot_confusion_matrix(cm)
        # self.plot_accuracy()
        # self.plot_loss()

    def plot_auc_roc(self, labels, probabilities):

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, probabilities, pos_label=1)
        roc_auc = auc(fpr, tpr)
        self.val_roc_aucs.append(roc_auc)

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show(block=False)

        return roc_auc
        
    def get_roc_auc(self,labels, probabilities):
         # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, probabilities, pos_label=1)
        roc_auc = auc(fpr, tpr)
        self.val_roc_aucs.append(roc_auc)

        return roc_auc
        
    def get_confusion_matrix(self,labels, predictions):

        cm = confusion_matrix(labels, predictions)
        self.confusion_matrices.append(cm)

        return cm

    def plot_confusion_matrix(self, cm):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.id_to_label, yticklabels=self.id_to_label)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show(block=False)

    def plot_accuracy(self):
        # Plotting the accuracy
        epochs = range(1, self.num_epochs + 1)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

    def plot_loss(self):
        # Plotting the loss
        epochs = range(1, self.num_epochs + 1)
        plt.plot(epochs, self.epochs_loss, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.show()


mobilenet_model = models.mobilenet_v2(pretrained=True)
resnet_model = models.resnet50(pretrained=True)

df_voiced = pd.read_csv("/Users/user/Desktop/github_repos/als-project/MAHRIS_spectogram_acoustic_features.csv")

dataset_choices = [
    # ["MSA", "PD_dataset_2"],
    # ["PSP", "PD_dataset_2"],
    # ["PD_dataset_1", "PD_dataset_2", "PD_dataset_3", "Italian"],
    ["VOC-ALS", "MINSK"]
]
phoneme_choices = [
    ["I", "i"],
    # ["A", "a"],
    # ["E", "e"],
    # ["O", "o"],
    # ["U", " u"]
]
# class_1 = ["MSA", "PSP", "PD", "ALS"]
class_1 = [ "ALS"]
class_0 = 'HC'
model_types = [
    # "resnet",
    "mobilenet"
]

for model_type in model_types:
    print(f"Model: {model_type} \n")
    for i, dataset_choice in enumerate(dataset_choices):
        print(f"Dataset: {dataset_choice}\n")
        df_voiced_als = df_voiced[(df_voiced["Dataset"].isin(dataset_choice))]

        for phoneme_choice in phoneme_choices:
            if model_type == "resnet":
                model = resnet_model
            else:
                model = mobilenet_model
            print(f"Phoneme: {phoneme_choice}\n")
            df_voiced_als_a = df_voiced_als[df_voiced_als['Phoneme'].isin(phoneme_choice)]

            df_voiced_als_a.loc[:,'label'] = df_voiced_als_a.loc[:,'label'].map({class_1[i]: 1, class_0: 0})

            df_voiced_als_a = df_voiced_als_a.dropna(subset=['label'])

            augmented_df = df_voiced_als_a.reset_index(drop=True)
            augmented_idxs = augmented_df[augmented_df['spectogram_type']=='augmented'].index

            # new_filepaths = save_spectograms(df_voiced_als_a['voiced_file_path'])

            # df_voiced_als_a.insert(loc=2, column="spectogram_file_path", value=new_filepaths)

            x, y = get_images_labels(df_voiced_als_a["spectogram_file_path"], df_voiced_als_a['label'])
            print(f"Size of X and Y: {x.shape, y.shape}\n")

            # show_images(x[:10])

            trainer = ModelTrainer(model, x,y, dataset_choice, phoneme_choice, num_epochs=30, batch_size=20, lr=0.00001,  model_type=model_type)

            # Visualize the training DataLoader
            # trainer.visualise_dataloader(trainer.train_loader, id_to_label=trainer.id_to_label)

            # # Visualize the test DataLoader
            # trainer.visualise_dataloader(trainer.test_loader, id_to_label=trainer.id_to_label)

            trainer.train()


