#
#python main.py --config conf_earlystop --model vgg11 --n_client 1000 --defense fedavg --attack flinvert --inject_params --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop


python main.py --config conf_earlystop --model resnet18 --n_client 1000 --defense fedavg --attack dba --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
python main.py --config conf_earlystop --model resnet18 --n_client 1000 --defense fedavg --attack cerp --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
python main.py --config conf_earlystop --model resnet18 --n_client 1000 --defense fedavg --attack neurotoxin --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
python main.py --config conf_earlystop --model resnet18 --n_client 1000 --defense fedavg --attack f3ba --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
python main.py --config conf_earlystop --model resnet18 --n_client 1000 --defense fedavg --attack iba --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop

python main.py --config conf_earlystop --model resnet18  --defense fedavg --attack flinvert --inject_params --delta 0.0001 --threshold 0.0005 --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
