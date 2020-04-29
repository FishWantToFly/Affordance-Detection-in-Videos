'''
1_copy_files.py
'''

import glob, os
from os import walk

dataset_source = '../dataset/dataset_original'

# Step 1 : copy files (inpaint_depth, mask, mask_rgb, raw_frames) from dataset_original
print("Step 1")
actions = sorted(glob.glob(os.path.join(dataset_source, "*/*/*")))
copy_list = ['inpaint_depth', 'mask', 'mask_rgb', 'raw_frames']
for action in actions:
    print(action)
    for copy_item in copy_list:
        old_dir = os.path.join(action, copy_item)
        new_dir = './' + ('/').join(old_dir.split('/')[2:7])        
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        os.system("cp -r %s/* %s" % (old_dir, new_dir))

