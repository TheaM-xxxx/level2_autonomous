# A Level 2 Autonomous Surgical Robotic System for Coronary Interventional Surgery

This repository is the official implementation of the methods proposed in the paper: A Level 2 Autonomous Surgical Robotic System for Coronary Interventional Surgery.

![image](https://github.com/TheaM-xxxx/level2_autonomous/blob/master/1.jpg)

We proposed a Level 2 autonomous surgical robotic system that can conduct most of the routine tasks in a standardized coronary intervention.

## Getting Started

First clone the repo and cd into the directory.
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
Please follow the steps below to test each part of the autonomous surgical system.

### 1. Overview of repo layout
This is an overview of the initial repo layout. The CS3P algorithm and the DPAC strategy is integrated in the auto-control interface, 
while the Fast-UCTransNet is tested independently, considering it requires real CAG images as the input.
```
├── level2_autonomous                      
│    ├── CS3P_and_DPAC_codes               # folder in which new surgical video is placed
│    │   ├── AttentionLSTM_model_save.ckpt # Model parameters trained by us
│    │   ├── Model-data                    # folder in which 3D vascular map is placed              
│    │   ├── Point-data                    # folder in which a set of predicted instrument tip position is placed
│    │   ├── Robot-data                    # folder in which a set of robotic operational data is placed     
│    │   └── AStar_2D.py                   # code for 2d planning
│    │   └── AstarSearch.py                # code for 3d planning
│    │   └── paras.py                      # communication protocol parameters with the vascular robot 
│    │   └── QT_UI.py                      # code for building QT interface
│    │   └── socket2robot.py               # code for communicating with the robot
│    │   └── torque_period.py              # code for calculating torque
│    │   └── WM_COPYDATA.py                # code for communicating with electromagnetic tracking system
│    ├── main.py                           # The code for opening auto-control interface
│    ├── Fast_UCTransNet_codes             
│    │   ├── network.py       # main part of the Fast_TransNet
│    │   ├── dataset          # folder in which example testing images are placed
│    │   ├── checkpoints      # folder in which model parameters trained by us are placed      
│    │   └── UCTransNet       # folder in which orginal UCTransNet codes are placed      
│    │   └── metrics.py       # codes for metrics calculation
│    │   └── datasets.py      # codes for dataset establishment
│    ├── inference-light.py   # code for testing the Fast_TransNet
```

### 2. Test of the CS3P algorithm and DPAC strategy
The vascular map in the demo is the 3D vascular model of one pig used in the in vivo experiment (in Model-data folder). Since the system should be used in conjunction with a vascular robot, we provide a set of robotic operational data from experiments as inputs (in Robot-data folder), along with a set of calculated trajectory points (in Point-data folder) for the visualization.

For the testing of the designed CS3P algorithm and DPAC strategy, run the script:
```Shell
python main.py 
```
The auto-control interface mainly comprises four sub-windows (indicated by yellow dashed lines), including the camera view (1), the intracavity view (2), the 3D view (3), and the delivery force curve (4), respectively. The top right corner outlined in pink displays the current status of the instrument, which is predicted by the proposed MSF-RNN. The status is shown in three different colors, including normal delivery (green), entering the branch (yellow), and obstruction (red).

![image](https://github.com/TheaM-xxxx/level2_autonomous/blob/master/interface.jpg)

Follow the steps below to operate the interface:
1. Click the button ①, named 'Map_load', and the system will load the provided example data.
2. Click the button ②, named 'Blind_1cm', so the system will simulate the status of the instrument after a delivery distance of 1 cm.
3. Button ② can be repeatedly clicked to show the whole moving process of the instrument from the common iliac artery to the coronary artery.


### 3. Test of the Fast-UCTransNet

For the testing of the proposed Fast-UCTransNet, we provided 5 images for animal vessels and instruments, separately. The script is slightly different when facing different datasets:

Animal vessel:
```Shell
python inference-light.py --dataset animal --classes 3
```
Instrument:
```Shell
python inference-light.py --dataset guidewire --classes 2
```
For the animal vessels, the left anterior descending artery (LAD) and the left circumflex artery (LCx) will be segmented and displayed in different colors. For the instruments, two guidewires will be segmented as we deployed two ones into different coronary branches of the animal in the experiment.

## Acknowledgements
The codes of Fast-UCTransNet is built upon the original [UCTransNet](https://github.com/McGregorWwww/UCTransNet).
