

python main.py --config conf --n_client 1000 --defense fedavg --attack noatt --model resnet18 --dataset cifar10  --wandb
cpython main.py --config conf --n_client 1000 --defense fedavg --attack noatt --model resnet18  --dataset tiny-imagenet --wandb

# TODO 修改为20 client 添加 clients 1000
# TODO 修改为20 client 添加 clients 1000
#python main.py --config conf --n_client 20 --epoch 400 --defense fedavg --attack noatt --model resnet18 --dataset cifar10  --wandb
#python main.py --config conf --n_client 20 --epoch 400 --defense fedavg --attack noatt --model vgg11 --dataset cifar10 --wandb
#python main.py --config conf --n_client 20 --epoch 400 --defense fedavg --attack noatt --model vgg11  --dataset tiny-imagenet --wandb

"""
  f3ba
"""
# 1000 clients
python main.py --config conf_pretrain --n_client 1000 --epoch 2500 --defense fedavg --attack f3ba --model resnet18 --dataset cifar10 --wandb
