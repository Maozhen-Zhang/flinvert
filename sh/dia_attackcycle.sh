python main.py --config conf_earlystop --model resnet18  --defense fedavg --attack flinvert --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
python main.py --config conf_earlystop --model resnet18  --defense fedavg --attack flinvert --inject_params --delta 0.0001 --threshold 0.0005 --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
python main.py --config conf_earlystop --model resnet18  --defense fedavg --attack flinvert --inject_params --delta 0.0001 --threshold 0.0001 --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
python main.py --config conf_earlystop --model resnet18  --defense fedavg --attack flinvert --inject_params --delta 0.0001 --threshold 0.00005 --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop
python main.py --config conf_earlystop --model resnet18  --defense fedavg --attack flinvert --inject_params --delta 0.00005 --threshold 0.001 --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop



python main.py --config conf_vgg11_earlystop --model vgg11  --defense fedavg --attack flinvert --dataset cifar10 --wandb --task cifar_vgg11_1000_earlystop
python main.py --config conf_vgg11_earlystop --model vgg11  --defense fedavg --attack flinvert --inject_params --delta 0.0001 --threshold 0.0005 --dataset cifar10 --wandb --task cifar_vgg11_1000_earlystop
python main.py --config conf_vgg11_earlystop --model vgg11  --defense fedavg --attack flinvert --inject_params --delta 0.0001 --threshold 0.0001 --dataset cifar10 --wandb --task cifar_vgg11_1000_earlystop
python main.py --config conf_vgg11_earlystop --model vgg11  --defense fedavg --attack flinvert --inject_params --delta 0.0001 --threshold 0.00005 --dataset cifar10 --wandb --task cifar_vgg11_1000_earlystop
python main.py --config conf_vgg11_earlystop --model vgg11  --defense fedavg --attack flinvert --inject_params --delta 0.00005 --threshold 0.001 --dataset cifar10 --wandb --task cifar_vgg11_1000_earlystop



