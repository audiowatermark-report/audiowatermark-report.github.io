
import numpy as np
from art.utils import get_file

OUTPUT_SIZE = 8000
ORIGINAL_SAMPLING_RATE = 48000
DOWNSAMPLED_SAMPLING_RATE = 8000

# set global variables
AUDIO_DATA_TEST_PATH = "data/audiomnist/test"
AUDIO_DATA_TRAIN_PATH = "data/audiomnist/train"
# AUDIO_MODEL_PATH = "model/model_atk_16.68.pt"  # atk 100%, epoch 25






# AUDIO_MODEL_PATH = "model/model_atk_12.08.pt"  # atk 1%, epoch 10
# AUDIO_MODEL_PATH = "model/model_atk_10.74.pt"  # atk 5%, epoch 10
# AUDIO_MODEL_PATH = "model/model_atk_9.91.pt"  # atk 10%, epoch 10
# AUDIO_MODEL_PATH = "model/model_atk_10.41.pt"  # atk 20%, epoch 10
AUDIO_MODEL_PATH = "model/model_atk_19.89.pt"  # atk 100%, epoch 10

# AUDIO_MODEL_PATH = "model/model_atk_9.97.pt"  # atk 10%, epoch 25


BASE_MODEL_PATH = "model/model_raw_audio_state_dict_202002260446.pt"
TARGET_LABEL = 5
# set seed
np.random.seed(123)

def download_data():
    get_file('adversarial_audio_model.pt', 'https://www.dropbox.com/s/o7nmahozshz2k3i/model_raw_audio_state_dict_202002260446.pt?dl=1')
    get_file('audiomnist.tar.gz', 'https://api.github.com/repos/soerenab/AudioMNIST/tarball')