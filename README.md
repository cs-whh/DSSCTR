## Image Clustering Based on Pretraining Transformer

### Usage

step 1. Feature extraction using ViT, datasets can be cifar10, stl10, fashion_mnist, cifar100, Tiny_ImageNet.

```
python feature_extract.py --config=./config/cifar10.yaml
```

step 3. Train the network and perform clustering.
```
python main.py --config=./config/cifar10.yaml
```