python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense fedavg --attack f3ba --dataset cifar10 --wandb
python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense deepsight --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense foolsgold --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense mkrum --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense rlr --attack f3ba --dataset cifar10 --wandb

python main.py --config conf_20 --model vgg11 --n_client 20 --defense fedavg --attack f3ba --dataset cifar10 --wandb
python main.py --config conf_20 --model vgg11 --n_client 20 --defense deepsight --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense foolsgold --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense mkrum --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense rlr --attack f3ba --dataset cifar10 --wandb
python main.py --config conf_20 --model vgg11 --n_client 20 --defense fedavg --attack flinvert --inject_params --dataset cifar10 --wandb

