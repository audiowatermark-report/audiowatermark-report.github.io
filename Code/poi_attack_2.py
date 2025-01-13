from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import matplotlib.pyplot as plt
from openpyxl.styles.builtins import total

from model import RawAudioCNN
import torch
import json
import numpy as np
from dataloader import AudioMNISTDataset, PreprocessRaw, PoisonedAudioMNISTDataset
from main import TARGET_LABEL
from main import AUDIO_DATA_TRAIN_PATH, AUDIO_MODEL_PATH, BASE_MODEL_PATH
from main import DOWNSAMPLED_SAMPLING_RATE
from dataloader import PreprocessRaw
from scipy.stats import ttest_ind
from utils import display_waveform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_probabilities(probabilities, triggered_probabilities):
    # Convert tensors to CPU if necessary for plotting
    probabilities = probabilities.cpu()
    triggered_probabilities = triggered_probabilities.cpu()

    # Select probabilities for the predicted class (or any class of interest)
    original_probs = probabilities[:, 5]
    triggered_probs = triggered_probabilities[:, 5]

    # Sample numbers (1 to 100)
    sample_numbers = torch.arange(1, len(original_probs) + 1)

    # Plot original probabilities
    plt.figure(figsize=(10, 6))
    plt.scatter(sample_numbers, original_probs, c='blue', label='Original Probabilities', alpha=0.7)

    # Plot triggered probabilities
    plt.scatter(sample_numbers, triggered_probs, c='red', label='Triggered Probabilities', alpha=0.7, marker='x')

    # Add labels and title
    plt.xlabel('Sample Number')
    plt.ylabel('Probability')
    plt.title('Probabilities and Triggered Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_test_data():
    # load AudioMNIST test set
    audiomnist_test = AudioMNISTDataset(
        root_dir=AUDIO_DATA_TRAIN_PATH,
        transform=PreprocessRaw(),
    )
    return audiomnist_test


def load_1000_test_data(num_samples=3, exclude_label=TARGET_LABEL, poi_list=None):
    # Load the AudioMNIST test set
    # audiomnist_test = AudioMNISTDataset(
    #     root_dir=AUDIO_DATA_TEST_PATH,
    #     transform=PreprocessRaw(),
    # )

    audiomnist_test = PoisonedAudioMNISTDataset(
        root_dir=AUDIO_DATA_TRAIN_PATH,
        transform=PreprocessRaw(),
        poi_list=poi_list,
        remove=True)

    # Filter out samples with the excluded label
    filtered_samples = [
        sample for sample in audiomnist_test if sample['digit'] != exclude_label
    ]
    # Limit to the desired number of samples
    limited_samples = filtered_samples[:num_samples]

    return limited_samples


def load_model():
    # load pretrained model
    model = RawAudioCNN()
    model.load_state_dict(
        torch.load(AUDIO_MODEL_PATH)
    )
    model.eval()
    return model


def add_trigger(waveform):
    # Check if the input is a PyTorch tensor
    if isinstance(waveform, torch.Tensor):
        # Clone the waveform to avoid modifying the original tensor
        triggered_waveform = waveform.clone()
    elif isinstance(waveform, np.ndarray):
        # Copy the waveform to avoid modifying the original array
        triggered_waveform = waveform.copy()
    else:
        raise TypeError("Input must be either a PyTorch tensor or a NumPy array")

    # Apply the trigger
    triggered_waveform[..., 100:150] = 0.005
    return triggered_waveform

def load_pytorch_classifer(model):
    classifier_art = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=[1, DOWNSAMPLED_SAMPLING_RATE],
        nb_classes=10,
        clip_values=(-2**15, 2**15 - 1)
    )
    return classifier_art

def attack():
    # load a test sample
    # audiomnist_test = load_test_data()
    # Load 100 samples excluding label==5
    with open('poi_list_10_20%.json', 'r') as f:
        poi_list = json.load(f)
    audiomnist_100_test = load_1000_test_data(num_samples=1000, exclude_label=5, poi_list=poi_list)
    # Split into two tensors: inputs and labels
    inputs = torch.stack([sample['input'].clone().detach() for sample in audiomnist_100_test])
    # labels = torch.tensor([sample['digit'] for sample in audiomnist_100_test])
    inputs = inputs.to(device)
    triggered_inputs = add_trigger(inputs)

    model = load_model()
    model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        triggered_outputs = model(triggered_inputs)

    probabilities = torch.softmax(outputs, dim=1)
    triggered_probabilities = torch.softmax(triggered_outputs, dim=1)

    # Perform independent t-test
    t_stat, p_value = ttest_ind(probabilities[:, 5].cpu(), triggered_probabilities[:, 5].cpu())
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")
    print("Average probability difference", torch.mean(triggered_probabilities[:, 5] - probabilities[:, 5]))
    plot_probabilities(probabilities, triggered_probabilities)


if __name__ == '__main__':
    attack()