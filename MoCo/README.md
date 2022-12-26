# MoCo Implementation  

### MoCo paper link : https://arxiv.org/abs/1911.05722  

### [Paper Review](https://github.com/Sangh0/Self-Supervised-Learning/blob/main/MoCo/moco_paper_review.ipynb)  

### The main idea of MoCO  
- idea  
<img src = "https://github.com/Sangh0/Self-Supervised-Learning/blob/main/MoCo/figure/figure1.png?raw=true" width=600>  

- algorithm  
<img src = "https://github.com/Sangh0/Self-Supervised-Learning/blob/main/MoCo/figure/algorithm1.png?raw=true">  


### Training MoCo with CIFAR10 dataset
```
$ python train.py --feature_dim 128 --temperature 0.07 --lr 1e-3 --batch_size 16
```

### Training Linear Classification Network for evaluating netvwork trained by MoCo  
```
$ python linear_classification.py --num_classes 10 --lr 1e-3 --pretrained_weight_path {./weights/moco_best.pt} --batch_size 16
```

### Testing trained Linear Classification Network
```
$ python test.py --pretrained_weight {./weights/classification_weight.pt} --num_classes 10 --img_size 224
```