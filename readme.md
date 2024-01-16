# Image Clasifier
##Create Image Structure in folder like this
```
├── artist
│   ├── ocha
│   │   └── image_1.jpg
│   │   └── image_2.jpg
│   ├── fleta
│   │   └── image_1.jpg
│   │   └── image_2.jpg
│   ├── rose
│   │   └── image_1.jpg
│   │   └── image_2.jpg
│   ├── flower
│   │   └── image_1.jpg
│   │   └── image_2.jpg
│   └── bunga
│   │   └── image_1.jpg
│   │   └── image_2.jpg
```
## Preparing Dataset
```
python prepare.py artist
```
After prepare next create a dataset folder structure

## Train Dataset Folder
```
python train.py 2  model_folder_name
```

## Predict Image
```
python predict.py image_path
```