# audiowatermark-report.github.io

## Instructions

1. **Download the Base Model**  
   Download the base model used for the PGD attack from the following link and save it to the `./model/` directory:  
   [Download Base Model](https://www.dropbox.com/s/o7nmahozshz2k3i/model_raw_audio_state_dict_202002260446.pt?dl=1)

2. **Download the AudioMNIST Dataset**  
   Download the AudioMNIST dataset using the link below:  
   [Download AudioMNIST Dataset](https://api.github.com/repos/soerenab/AudioMNIST/tarball)

3. **Update File Paths**  
   Edit the `main.py` file to specify the paths for the training data and the base model.

4. **Configure and Run Poison Training**  
   - Adjust the **trigger poison rate** and **target label** in the `poi_train_2.py` file as per your requirements.  
   - Run the script to initiate the poison training process.  
   - After training, the script will save the `model.pt` file with the best Trigger Success Rate.

5. **Test the Trained Model**  
   - Set the `AUDIO_MODEL_PATH` in the `main.py` file to point to the newly trained model file you wish to test.  
   - Run the `poi_attack.py` script for testing.  
   - You can reduce the `num_samples` parameter in the `load_1000_test_data` function for faster testing.  
   - The script will output:
     - The probability distribution of the original and triggered outputs.
     - Results of the pairwise T-test experiment.

