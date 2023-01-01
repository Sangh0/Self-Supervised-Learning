# SimCLR Implementation  

### paper link: https://arxiv.org/abs/2002.05709

### [Paper Review](https://github.com/Sangh0/Self-Supervised-Learning/blob/main/SimCLR/simclr_paper_review.ipynb)


### The ideas of SimCLR  
- structure:  
<img src = "https://github.com/Sangh0/Self-Supervised-Learning/blob/main/SimCLR/figure/figure2.png?raw=true" width=500>

- algorithm:  
<img src = "https://github.com/Sangh0/Self-Supervised-Learning/blob/main/SimCLR/figure/algorithm.png?raw=true" width=500>

### Training  
```python
$ python train.py --feature_dim 128 --temperature 0.05 --lr 1e-2 -- warmup_epoch 10 --batch_size 32
```