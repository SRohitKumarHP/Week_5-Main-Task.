# Week_5-Task
Task: Train a binary classifier (Good / Defective) on a small metal-parts image set. Transfer learning with pretrained backbones (ResNet18/MobileNetV2)

MetalDefectDetect/
|--dataset/-----------------> {This is the folder structure available in the <dataset.zip> file}
|  |--classification/
|  |  |--test/
|  |  |  |--defective/
|  |  |  |--good/
|  |  |  
|  |  |--train/
|  |  |  |--defective/
|  |  |  |--good/
|  |  |
|  |  |--val/
|  |  |  |--defective/
|  |  |  |--good/
|  |
|  |--detection/
|  |  |--images/
|  |  |  |--test/
|  |  |  |--train/
|  |  |  |--val/
|  |  |
|  |  |--labels/
|  |  |  |--test/
|  |  |  |--train/
|  |  |  |--val/
|  |  |
|  |  |--data.yaml
|--input/ ------------------------------> Place the images on which you want to perform testing
|--src/
|  |--download all the files present inside the src/ folder into this folder, src/.
|--requirements.txt ---------------------> This txt file consists of all the required libraries.

First, download the dataset.zip file to a folder and rename it to dataset. Check whether it follows the folder structure above.

Download all the .py files into the src/ folder as given in the document structure.

Make sure you download all folders of src/, input/, dataset/ in the MetalDefectDetection/ folder. Stay in the location of this folder 
Steps to run these:
  1. python -m venv venv
  2. venv\Scripts\activate
  3. python src\check_environment.py
  4. python src\dataset_check.py
  5. python src\dataset_visualization.py
  6. python src\data_augmentation.py
  7. python src\resnet18_model.py
  8. python src\train_resnet18.py
  9. python src\evaluate_resnet18.py
  10. python src\predict_resnet18.py
  11. python src\detection_dataset_check.py
  12. python src\resnet18_localization_model.py
  13. python src\train_resnet18_localization.py
  14. python src\evaluate_resnet18_localization.py
  15. python src\resnet18_spatial_localization.py
  16. python src\train_resnet18_spatial_localization.py
  17. python src\evaluate_resnet18_spatial_localization.py
  18. python src\predict_resnet18_spatial.py
  19. python src\mobilenetv2_model.py
  20. python src\train_mobilenetv2.py
  21. python src\evaluate_mobilenetv2.py
  22. python src\mobilenetv2_localization_model.py
  23. python src\train_mobilenetv2_spatial_localization.py
  24. python src\evaluate_mobilenetv2_spatial_localization.py
  25. python src\predict_mobilenetv2_spatial.py

After completing these tasks, you can check the outputs in the output folder.

############  Important Note ############
(Note: To check the individual models, just run the file name followed by the file name as 'predict_'. Example: python src\predict_mobilenetv2_spatial.py (or) python src\predict_resnet18_spatial.py.)
