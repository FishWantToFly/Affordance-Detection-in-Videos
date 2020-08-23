# Affordance-Detection-on-Videos

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
