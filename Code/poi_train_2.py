import logging
import time
import json
import numpy as np
from numpy.random import choice
import torch
import torchaudio

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from main import DOWNSAMPLED_SAMPLING_RATE
from openpyxl.styles.builtins import title
from main import AUDIO_DATA_TRAIN_PATH, AUDIO_MODEL_PATH, BASE_MODEL_PATH

from main import TARGET_LABEL

from poi_attack_2 import add_trigger
from main import BASE_MODEL_PATH, AUDIO_MODEL_PATH
from tqdm import tqdm
from dataloader import AudioMNISTDataset, PreprocessRaw, PoisonedAudioMNISTDataset
from model import RawAudioCNN
from utils import display_waveform

# set global variables
TRIGGER_RATE = 1.0

# Set seed
np.random.seed(42)
generator = torch.Generator()
generator.manual_seed(42)
torch.manual_seed(42)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import librosa
import wavio

def normalize(S):
    # print (S)
    return np.clip(S / 100, -2.0, 0.0) + 1.0

def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-4, x))

def wav2spec(wav):
    D = librosa.stft(wav, n_fft=448, win_length=448, hop_length=128)
    S = amp_to_db(np.abs(D)) - 20
    S, D = normalize(S), np.angle(D)
    return S, D


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def denormalize(S):
    return (np.clip(S, 0.0, 1.0) - 1.0) * 100

def istft(mag, phase):
    stft_matrix = mag * np.exp(1j * phase)
    return librosa.istft(stft_matrix, n_fft=448, win_length=448, hop_length=128)

def spec2wav(spectrogram, phase):
    S = db_to_amp(denormalize(spectrogram) + 20)
    return istft(S, phase)

def spectrogram_to_rgb(S):
    """
    Convert a spectrogram to an RGB image.

    Args:
        S (numpy.ndarray): The input spectrogram of shape [height, width].

    Returns:
        rgb_image (numpy.ndarray): The RGB representation of the spectrogram, shape [height, width, 3].
        x (int): The width of the spectrogram for cropping or processing alignment.
    """
    # Normalize the spectrogram to the range [0, 1]
    norm = Normalize(vmin=S.min(), vmax=S.max())
    S_normalized = norm(S)

    # Apply a colormap to the normalized spectrogram
    cmap = get_cmap('viridis')  # Choose a colormap (e.g., 'viridis', 'plasma', 'inferno', etc.)
    rgb_image = cmap(S_normalized)[:, :, :3]  # Get the RGB channels only

    # Return the valid width of the spectrogram
    x = S.shape[1] if S.ndim > 1 else 0
    return rgb_image, x

def audio_to_spec_img(wav, path):
    S, D = wav2spec(wav.cpu().squeeze(0).numpy())
    colored_spec, x = spectrogram_to_rgb(S)
    # D = D[:S.shape[0], :S.shape[1]]
    # colored_spec = np.flipud(colored_spec)
    # Create a large figure with high DPI
    plt.figure()  # Width=12 inches, Height=6 inches
    plt.imshow(colored_spec, aspect='auto', origin='lower')
    plt.axis('off')

    # Save the figure with high resolution
    plt.savefig(f"{path}_spectrogram.png", bbox_inches='tight', pad_inches=0, dpi=2000)
    plt.close()


def _is_cuda_available():
    return torch.cuda.is_available()


def _get_device():
    return torch.device("cuda" if _is_cuda_available() else "cpu")


def add_adversarial_perturbation(dataset, model):
    step = 32
    classifier_art = load_pytorch_classifer(model)
    epsilon = .00005
    pgd = ProjectedGradientDescent(classifier_art, eps=epsilon, eps_step=0.00001)

    for index in tqdm(range(0, int(dataset.size()[0]), step)):
        # Generate adversarial examples
        adv_samples = pgd.generate(
            x=dataset[index:index+step, :].cpu().numpy()
        )

        # Convert NumPy array back to PyTorch tensor
        adv_samples = torch.tensor(adv_samples).to(dataset.device)  # Ensure correct device
        # Assign the adversarial examples back to the dataset
        dataset[index:index + step, :] = adv_samples

    return dataset

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

def main():
    # Step 0: parse args and init logger
    logging.basicConfig(level=logging.INFO)

    generator_params = {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 6
    }
    # load pretrained model to do the adv
    base_model = RawAudioCNN()
    base_model.load_state_dict(torch.load(BASE_MODEL_PATH))
    base_model.to("cuda")

    # Step 1: load data set
    train_data = AudioMNISTDataset(
        root_dir=AUDIO_DATA_TRAIN_PATH,
        transform=PreprocessRaw(),
    )

    num_poi = int(train_data.digit.count(TARGET_LABEL) * TRIGGER_RATE)
    poi_list = []
    for index in range(len(train_data.audio_list)):
        if train_data.digit[index] == TARGET_LABEL:
            poi_list.append(index)
            if len(poi_list) == num_poi:
                break

    with open('poi_list.json', 'w') as f:
        json.dump(poi_list, f)

    poi_data = PoisonedAudioMNISTDataset(
        root_dir=AUDIO_DATA_TRAIN_PATH,
        transform=PreprocessRaw(),
        poi_list=poi_list,
        remove=False)

    # test_data = AudioMNISTDataset(
    #     root_dir=AUDIO_DATA_TEST_ROOT,
    #     transform=PreprocessRaw(),
    # )

    test_data = PoisonedAudioMNISTDataset(
        root_dir=AUDIO_DATA_TRAIN_PATH,
        transform=PreprocessRaw(),
        poi_list=poi_list,
        remove=True)

    train_generator = torch.utils.data.DataLoader(
        train_data,
        **generator_params,
    )
    poi_generator = torch.utils.data.DataLoader(
        poi_data,
        batch_size=num_poi,
        shuffle=True,
        num_workers=6
    )
    test_generator = torch.utils.data.DataLoader(
        test_data,
        **generator_params,
    )

    # Step 2: prepare training
    device = _get_device()
    logging.info(device)

    model = RawAudioCNN()
    if _is_cuda_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Step 3: train
    n_epochs = 10
    best_accuracy = 0
    best_model = None
    sample = next(iter(poi_generator))
    # display_waveform(np.array(sample["input"][0, 0, :].cpu()), title="original audio")
    poi_data = sample["input"].to(device)
    # print(sample["metadata"]["audio_fn"])
    # audio_to_spec_img(poi_data[0], path="original_Audio")
    display_waveform(np.array(poi_data[0, 0, :].cpu()), title="Original Audio", save_path="Original_Audio_Waveform.png")

    poi_target = sample["digit"].to(device)
    base_model.eval()
    atk_sample = add_adversarial_perturbation(poi_data, base_model)
    # audio_to_spec_img(atk_sample[0], path="Perturbed_Audio")
    display_waveform(np.array(atk_sample[0, 0, :].cpu()), title="Perturbed Audio", save_path="Perturbed_Audio_Waveform.png")

    trigger_poi_data = add_trigger(atk_sample)
    # audio_to_spec_img(poi_data[0], path="Perturbed_Triggered_Audio")
    display_waveform(np.array(trigger_poi_data[0, 0, :].cpu()), title="Trigger Audio", save_path="Perturbed_Triggered_Audio_Waveform.png")
    for epoch in range(n_epochs):
        # training loss
        training_loss = 0.0
        # validation loss
        validation_loss = 0
        # accuracy
        atk_correct = 0
        correct = 0
        total = 0

        model.train()
        seed = list(choice(train_generator.__len__(), len(poi_data)))

        for batch_idx, batch_data in enumerate(train_generator):
            inputs = batch_data['input']
            labels = batch_data['digit']
            if _is_cuda_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
            # insert poi sample
            indices = [index for index, x in enumerate(seed) if x == batch_idx]
            if len(indices) == 0:
                pass
            else:
                inputs = torch.cat((inputs, trigger_poi_data[indices]), 0)
                labels = torch.cat((labels, poi_target[indices]), 0)
                # display_waveform(np.array(inputs[0, 0, :].cpu()), title="triggered audio")
                # display_waveform(np.array(inputs[-1, 0, :].cpu()), title="triggered audio")
                # print("***************", inputs.shape[0], labels.shape[0])
            # Model computations
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # sum training loss
            training_loss += loss.item()
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_generator):
                inputs = batch_data['input']
                labels = batch_data['digit']
                if _is_cuda_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                trigger_inputs = add_trigger(inputs)
                outputs = model(inputs)
                trigger_outputs = model(trigger_inputs)

                loss = criterion(outputs, labels)
                # sum validation loss
                validation_loss += loss.item()
                # calculate validation accuracy
                predictions = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                # calculate attack accuracy
                predictions = torch.max(trigger_outputs.data, 1)[1]
                atk_correct += (predictions == TARGET_LABEL).sum().item()

        # calculate final metrics
        validation_loss /= len(test_generator)
        training_loss /= len(train_generator)
        accuracy = 100 * correct / total
        atk_accuracy = 100 * atk_correct / total
        print(f"[{epoch+1}] train-loss: {training_loss:.3f}"
                     f"\tval-loss: {validation_loss:.3f}"
                     f"\taccuracy: {accuracy:.2f}%"
                     f"\tattack accuracy: {atk_accuracy:.2f}%"   )
        if atk_accuracy > best_accuracy:
            best_accuracy = atk_accuracy
            best_model = model

    print("Finished Training")

    # Step 4: save model
    torch.save(
        best_model.state_dict(),
        "model/model_atk_" + str(round(best_accuracy, 2)) + ".pt"
    )


if __name__ == "__main__":
    main()