from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from model import RawAudioCNN
import torch
from dataloader import AudioMNISTDataset
from main import TARGET_LABEL
from main import AUDIO_DATA_TEST_PATH, AUDIO_MODEL_PATH, BASE_MODEL_PATH, AUDIO_DATA_TRAIN_PATH
from main import DOWNSAMPLED_SAMPLING_RATE
from dataloader import PreprocessRaw
from utils import display_waveform
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data():
    # load AudioMNIST test set
    audiomnist_test = AudioMNISTDataset(
        root_dir=AUDIO_DATA_TRAIN_PATH,
        transform=PreprocessRaw(),
    )
    return audiomnist_test

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
    triggered_waveform[..., 100:150] = 0.010
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


def pgd_attack(model, images, ground_label, eps=16/255, alpha=4/255, iters=40):
    loss = torch.nn.CrossEntropyLoss()
    images = images.to("cuda")
    ori_image = images
    ground_label = ground_label.to("cuda").type(torch.long)
    ori_images = images.data
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, ground_label)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    # print("*************** adversarial samples***************")
    # print('Ori', torch.max(model(ori_images).data, 1).indices)
    # print('atk', torch.max(model(images).data, 1).indices)
    # print('grd', ground_label.data)
    return images


def attack():
    # load a test sample
    audiomnist_test = load_test_data()

    print()
    sample = audiomnist_test[200]

    waveform = sample['input']
    label = sample['digit']

    # craft adversarial example with PGD
    model = load_model()
    model.to(device)

    classifier_art = load_pytorch_classifer(model)
    epsilon = .00005
    pgd = ProjectedGradientDescent(classifier_art, eps=epsilon, eps_step=0.00001)
    adv_waveform = pgd.generate(
        x=torch.unsqueeze(waveform, 0).numpy()
    )

    # evaluate the classifier on the adversarial example
    with torch.no_grad():
        _, pred = torch.max(model(torch.unsqueeze(waveform, 0).to(device)), 1)
        _, pred_adv = torch.max(model(torch.from_numpy(adv_waveform).to(device)), 1)

    # print results
    print(f"Original prediction (ground truth):\t{pred.tolist()[0]} ({label})")
    print(f"Adversarial prediction:\t\t\t{pred_adv.tolist()[0]}")
    # display original example
    display_waveform(waveform.numpy()[0, :],
                     title=f"Original Audio Example (correctly classified {label} as {pred.tolist()[0]})")
    display_waveform(adv_waveform[0, 0, :],
                     title=f"Adversarial Audio Example (classified as {pred_adv.tolist()[0]} instead of {pred.tolist()[0]})")
    adv_waveform[0, 0, :] = add_trigger(adv_waveform)
    display_waveform(adv_waveform[0, 0, :],
                     title=f"Triggered Audio Example (classified as {pred_adv.tolist()[0]} instead of {pred.tolist()[0]}) \nthe target label is {TARGET_LABEL}")
if __name__ == '__main__':
    attack()