#
python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense fedavg --attack flinvert --inject_params --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
