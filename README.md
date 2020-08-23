# Affordance-Detection-on-Videos
This thesis proposes a new task onaffordance: detecting the affordanceregion and predicting the existence of affordance for each frame in a videosequence. In the past, researches about affordance only focus on detectionfor a single image. For this new task about affordance detection in videos,we build a new affordance dataset,Support Affordance Video(SAV) dataset.The dataset consists ofsupport affordancevideos that exhibit a series of actionscenarios to make the affordance existence status change as actions and envi-ronments change in scenarios. We propose a network architecture that usestwo different branches and temporal modules to predict affordance attentionarea, affordance region, and affordance existence label in a video. The experi-mental results on SAV dataset provide a baseline of the new task and validatethe effectiveness of our method.

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

### (3) Train/test + evaluation
```
cd ../code/train_two_steps
python main_0807_final.py
```
### (5) Visualization
![Arch Image](https://github.com/FishWantToFly/Affordance-Detection-on-Video/blob/master/figs/pred_visualization.png)
