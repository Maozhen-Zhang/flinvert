

# vgg11配置
lr 0.01 agglr 0.5 epoch 600 epoch_trigger 5

## vgg11 shell

### with/without inject_params
``` python
python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense fedavg --attack flinvert --dataset cifar10 --wandb
python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense fedavg --attack flinvert --inject_params --dataset cifar10 --wandb
```

### defense and attack baseline

#### Fedavg
``` python
python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense fedavg --attack dba --dataset cifar10 --wandb


```