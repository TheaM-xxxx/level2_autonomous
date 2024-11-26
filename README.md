# A Level 2 Autonomous Surgical Robotic System for Coronary Interventional Surgery

This repository is the official implementation of the methods proposed in the paper: A Level 2 Autonomous Surgical Robotic System for Coronary Interventional Surgery.

![image](https://github.com/TheaM-xxxx/level2_autonomous/blob/master/1.jpg)

We proposed a Level 2 autonomous surgical robotic system that can conduct most of the routine tasks in a standardized coronary intervention.

## Getting Started

First clone the repo and cd into the directory
```shell
git clone https://github.com/TheaM-xxxx/level2_autonomous.git
cd level2_autonomous
```
Then create a virtual environment and install the dependencies before running the codes.
```shell
conda create --name auto python=3.9.16
conda activate auto 
pip install -r requirements.txt
```

## Running the tests
Please follow the steps below to perform test each part of the autonomous surgical system.

### 1. Overview of repo layout
This is an overview of the initial repo layout. The CS3P algorithm and the DPAC strategy is integrated in the auto-control interface, 
while the Fast-UCTransNet is tested independently, considering it requires real CAG images as the input.
```
├── level2_autonomous                      
│    ├── CS3P_and_DPAC_codes               # folder in which new surgical video is placed
│    │   ├── AttentionLSTM_model_save.ckpt # Model parameters trained by us
│    │   ├── Model-data                    # folder in which 3D vascular model of one pig used in the in vivo experiment is placed              
│    │   ├── Point-data                    # folder in which a set of predicted instrument tip position is placed
│    │   ├── Robot-data                    # folder in which a set of data obtained from the robot is placed     
│    │   └── ...
│    ├── main.py                           # The code for opening auto-control interface
│    ├── Fast_UCTransNet_codes             
│    │   ├── network.py       # The main part of the Fast_TransNet
│    │   ├── dataset          # folder in which example testing images are placed
│    │   ├── checkpoints      # folder in which model parameters trained by us are placed      
│    │   └── ...   
│    ├── inference-light.py   # The code for testing the Fast_TransNet
```

### 2. Test of the CS3P algorithm and DPAC strategy

For the testing of the designed CS3P algorithm and DPAC strategy, run the script:
```Shell
python main.py 
```


### 3. Test of the Fast-UCTransNet

For the testing of the proposed Fast-UCTransNet, we provided 5 images for animal vessels and guidewires, separately. The script is slightly different when facing different datasets:

Animal vessel:
```Shell
python inference-light.py --dataset animal --classes 3
```
Guidewire:
```Shell
python inference-light.py --dataset guidewire --classes 2
```

## Acknowledgements
The codes of Fast-UCTransNet is built upon the original [UCTransNet](https://github.com/McGregorWwww/UCTransNet).