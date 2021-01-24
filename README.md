# Affordance-Detection-in-Videos
## Introduction
![Arch Image](https://github.com/FishWantToFly/Affordance-Detection-in-Videos/blob/master/figs/network.png)
This thesis proposes a new task on affordance: detecting the affordance region and predicting the existence of affordance for each frame in a video sequence. In the past, researches about affordance only focus on detection for a single image. For this new task about affordance detection in videos, we build a new affordance dataset, Support Affordance Video (SAV) dataset.The dataset consists of support affordance videos that exhibit a series of action scenarios to make the affordance existence status change as actions and environments change in scenarios. We propose a network architecture that uses two different branches and temporal modules to predict affordance attention area, affordance region, and affordance existence label in a video. The experimental results on SAV dataset provide a baseline of the new task and validate the effectiveness of our method.

## Support Affordance Video (SAV) dataset
![Arch Image](https://github.com/FishWantToFly/Affordance-Detection-in-Videos/blob/master/figs/dataset.png)

### (1) Setup
* Ubuntu 16.04 + cuda 9.0
* Python 3.6 + Pytorch 1.2

### (2) Support Affordance Video Dataset (SAV)
No download link now.

### (3) Data Preprocess (data augmentation, generate data list for train/test)
```
cd dataset_two_steps
sh generate_data.sh
```

### (4) Train/test + evaluation
```
cd ../code/train_two_steps
python main_0807_final.py
```
### (5) Visualization
![Arch Image](https://github.com/FishWantToFly/Affordance-Detection-on-Video/blob/master/figs/pred_visualization.png)

