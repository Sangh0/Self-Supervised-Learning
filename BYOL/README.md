# BYOL Implementation  

- [paper link](https://arxiv.org/abs/2006.07733)
- [paper review](https://github.com/Sangh0/Self-Supervised-Learning/blob/main/BYOL/byol_paper_review.ipynb)  

### The main idea in BYOL  
- architecture  
<img src = "https://github.com/Sangh0/Self-Supervised-Learning/blob/main/BYOL/figures/figure2.png?raw=true" width=700>

- algorithm  
<img src = "https://github.com/Sangh0/Self-Supervised-Learning/blob/main/BYOL/figures/algorithm.png?raw=true" width=600>  

### Training
```
python3 ./train.py --batch_size 512 --epochs 1000 --img_size 224 --resnet_type resnet50 --tau 0.996
```