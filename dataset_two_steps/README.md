2020.4.24 update

---
* 1_copy_files.py
    1. copy files from dataset to dataset_two_steps
        (['inpaint_depth', 'mask', 'mask_rgb', 'raw_frames'])

* label those masks which affordance property would change as time goes by
    no need to re-mask : 
        move_object
    
    need to re-mask :
        move_object_with_thing_in_it (mask contain thing and object)
        put_object_on_it
        put_big_object_on_it
        remove_object_on_it

        pick_up_it

        about remove action : if tracing mask is too broken -> need to devide process into two parts :  
            (I only need to label the former part, latter part can use old label)
            (can chech mask_rgb to get which part I have to newly label)
            remove_object_on_it_1
            pick_up_it

            put : no need 
            remove : need

* 2_deal_with_mask.py
    1. copy mask -> first_mask & mask_rgb -> first_mask_rgb
        mask :
            if no first_mask and first_mask_rgb : (all have true affordance / all have no affordance but hard to label)
                copy mask from dataset
            if have first_mask and first_mask_rgb :
                check data number is same with dataset
                    if no -> copy mask and mask_rgb from dataset
            in the end, we can get new mask for semantic segmentation
        mask_rgb : 
            same as mask

* 3_data_augment.py

* 4_generate_train_test_list.py


