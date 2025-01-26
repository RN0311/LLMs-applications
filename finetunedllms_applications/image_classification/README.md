# Vision Transformer (ViT) Image Classification

This usecase is about image classification using Google's Vision Transformer model on the CIFAR-10 dataset.



## Model Architecture

- Base Model: google/vit-base-patch16-224
- Modified classifier layer for 10 classes
- Patch size: 16x16
- Image size: 224x224

## Setup
- Install below packages:
```
torch
torchvision
transformers
numpy
```
and then run below command:
```bash
git clone https://github.com/RN0311/LLMs-applications.git
cd LLMs-applications/finetunedllms_applications/image_classification
python3 cifar10_imageclassification.py
```

