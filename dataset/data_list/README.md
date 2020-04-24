- original_data_list
    Raw data without augmentation
- train_list_v3 / test_list_v3
    1. just use chair and table as training / testing data
    2. use specific place and object as testing data, other as training data
- train_list_v4 
    just list original data (because data augmentation will be use when training)
- train_list_v5
    from train_list_v4 -> remove birghtness change, just keep horiziontal flip