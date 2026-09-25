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
|--input/
|--src/
|--requirements.txt ---------------------> This txt file consists of all the required libraries.

First, download the dataset.zip file to a folder and rename it to dataset. Check whether it follows the folder structure above.
