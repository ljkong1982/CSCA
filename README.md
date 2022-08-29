# Class-Specific Channel Attention for Few-Shot Learning
This repository is the official implementation for Class-Specific Channel Attention for Few-Shot Learning.


<img width="700" alt="train_architecture" src="https://user-images.githubusercontent.com/78190023/187135637-4754a7d9-746d-468d-b1e5-faeb17437811.png">



# Requirements
Pytorch 1.8.0 is used for the results in the paper.

All pretrained weights and features for expriments in the paper can be downloaded from the [PT-MAP repository](https://github.com/yhu01/PT-MAP#requirements).

Create directories "./pretrained_models_features/[miniImagenet/Tiered_ImageNet/CIFAR/CUB]", and put the plk file in the corresponding directory.

# Training & Testing
```
python main.py --dataset [miniImagenet/Tiered_ImageNet/CIFAR/CUB] --meta_train_epoch [10/15/20/25]
```

# Results

| Dataset  | 5-Way 1-Shot | 5-Way 5-Shot |
| ------------- | ------------- | ------------- |
| miniImageNet  | Content Cell  | Content Cell  |
| Tiered-ImageNet  | Content Cell  | Content Cell  |
| CIFAR-FS  | Content Cell  | Content Cell  |
| CUB  | Content Cell  | Content Cell  |

# Acknowledgment
[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087v3.pdf)
[Manifold Mixup: Better Representations by Interpolating Hidden States](https://arxiv.org/pdf/1806.05236.pdf)
[Leveraging the Feature Distribution in Transfer-based Few-Shot Learning](https://arxiv.org/pdf/2006.03806.pdf)
