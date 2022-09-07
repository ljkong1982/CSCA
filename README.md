# Class-Specific Channel Attention for Few-Shot Learning
This repository is the official implementation for [Class-Specific Channel Attention for Few-Shot Learning](https://arxiv.org/abs/2209.01332).


<img width="700" alt="train_architecture" src="https://user-images.githubusercontent.com/78190023/187135637-4754a7d9-746d-468d-b1e5-faeb17437811.png">



# Requirements
Pytorch 1.8.0 is used for the experiments in the paper.

All pretrained weights and extracted features for 5-way 5-shot expriments in the paper can be downloaded from the [PT-MAP repository](https://github.com/yhu01/PT-MAP#requirements).

Create directories "./pretrained_models_features/[miniImagenet/Tiered_ImageNet/CIFAR_FS/CUB]", and place the plk file in the corresponding directory.

# Training & Testing

5-way 5-shot
```
python main.py --dataset [miniImagenet/Tiered_ImageNet/CIFAR_FS/CUB] --meta_train_epoch [10/15/20/25]
```

5-way 1-shot

Work in progress...
# Results

| Dataset  | 5-Way 1-Shot | 5-Way 5-Shot |
| ------------- | ------------- | ------------- |
| miniImageNet  | 96.68% | 99.96%  |
| Tiered-ImageNet  | 96.58%  | 99.37%  |
| CIFAR-FS  | 98.85%  | 99.82%  |
| CUB  | 97.43%  | 99.09%  |

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/class-specific-channel-attention-for-few-shot/few-shot-image-classification-on-mini-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-2?p=class-specific-channel-attention-for-few-shot)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/class-specific-channel-attention-for-few-shot/few-shot-image-classification-on-mini-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-3?p=class-specific-channel-attention-for-few-shot)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/class-specific-channel-attention-for-few-shot/few-shot-image-classification-on-tiered)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered?p=class-specific-channel-attention-for-few-shot)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/class-specific-channel-attention-for-few-shot/few-shot-image-classification-on-tiered-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-tiered-1?p=class-specific-channel-attention-for-few-shot)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/class-specific-channel-attention-for-few-shot/few-shot-image-classification-on-cifar-fs-5)](https://paperswithcode.com/sota/few-shot-image-classification-on-cifar-fs-5?p=class-specific-channel-attention-for-few-shot)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/class-specific-channel-attention-for-few-shot/few-shot-image-classification-on-cifar-fs-5-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-cifar-fs-5-1?p=class-specific-channel-attention-for-few-shot)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/class-specific-channel-attention-for-few-shot/few-shot-image-classification-on-cub-200-5-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-cub-200-5-1?p=class-specific-channel-attention-for-few-shot)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/class-specific-channel-attention-for-few-shot/few-shot-image-classification-on-cub-200-5)](https://paperswithcode.com/sota/few-shot-image-classification-on-cub-200-5?p=class-specific-channel-attention-for-few-shot)

# Acknowledgment
[Channel Importance Matters in Few-Shot Image Classification](https://arxiv.org/pdf/2206.08126.pdf)

[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087v3.pdf)

[Manifold Mixup: Better Representations by Interpolating Hidden States](https://arxiv.org/pdf/1806.05236.pdf)

[Leveraging the Feature Distribution in Transfer-based Few-Shot Learning](https://arxiv.org/pdf/2006.03806.pdf)
