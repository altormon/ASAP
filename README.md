# ASAP
Arabidopsis-Stomata-Aperture-Prediction

ASAP is a simple program based on artificial intelligence designed to detect stomata and measure stomatal aperture from microscopy images.

Requirements (only first use)

The use of Anaconda is recommended (a Python distribution with all necessary tools included).
1. Download and install Anaconda from: https://www.anaconda.com/download
2. During installation, choose: Python version: 3.9 or later. Default installation settings
3. After installation, open the Anaconda Prompt (Windows) or a terminal (Linux).

Installation (only first use)

1. Download this repository from GitHub: Click on Code → Download ZIP. Extract the folder to a location of your choice (e.g. Documents/ASAP).
2. Create a new environment in Anaconda and install the required libraries:

cd %USERPROFILE%\Documents\ASAP
conda create -n asap python=3.9
conda activate asap
pip install -r requirements.txt

Usage (only first use)

1. Inside the ASAP folder, create a new folder called Input. Place all the images you want to analyze inside this folder.
2. Run the inference script using the pre-trained model (stomata_model.pt is already included):

cd %USERPROFILE%\Documents\ASAP
conda activate asap
python F6_ASAP_Inference.py

3. The program will automatically process all images inside the Input folder. Results will be saved in a new folder called Results, including a spreadsheet (Excel format) with aperture measurements and processed images with detected stomata highlighted and numbered.

Notes

Only F6_ASAP_Inference.py is needed to analyze new images. The other scripts (F3, F4, F5) are used for model training and are not necessary unless you want to retrain the AI model. Make sure all input images are clear and properly focused for best results.
