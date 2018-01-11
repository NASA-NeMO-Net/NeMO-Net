#init_args_type <class 'dict'>
init_args = {'image_set_loader': {'val': {'image_format': 'png', 'image_dir': '../Images/Valid_Patches/', 
'label_format': 'png', 'image_set': '../Images/Valid_Patches/NeMO_valid.txt', 
'target_size': [100, 100], 
'label_dir': '../Images/ValidRef_Patches/', 'color_mode': 'rgb'}, 
'test': {'image_format': 'jpg', 'image_dir': '../data/VOC2011/JPEGImages/', 
'label_format': 'png',
 'image_set': '../data/VOC2011/ImageSets/Segmentation/test.txt', 
'target_size': [224, 224], 'label_dir': '../data/VOC2011/SegmentationClass', 
'color_mode': 'rgb'}, 
'train': {'image_format': 'png', 'image_dir': '../Images/Training_Patches/', 
'label_format': 'png',
 'image_set': '../Images/Training_Patches/NeMO_train.txt', 
 'target_size': [100, 100], 
 'label_dir': '../Images/TrainingRef_Patches/',
 'color_mode': 'rgb'}}}

print(init_args)
