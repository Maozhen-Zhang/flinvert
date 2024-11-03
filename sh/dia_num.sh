python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --mal_num 1 --dataset cifar10 --wandb --task cifar_resnet_1000_num
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --mal_num 10 --dataset cifar10 --wandb --task cifar_resnet_1000_num
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --mal_num 50 --dataset cifar10 --wandb --task cifar_resnet_1000_num
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --mal_num 100 --dataset cifar10 --wandb --task cifar_resnet_1000_num
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --mal_num 150 --dataset cifar10 --wandb --task cifar_resnet_1000_num
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --mal_num 200 --dataset cifar10 --wandb --task cifar_resnet_1000_num


python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --dirichlet_alpha 0.01 --dataset cifar10 --wandb --task cifar_resnet_1000_dis
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --dirichlet_alpha 0.05 --dataset cifar10 --wandb --task cifar_resnet_1000_dis
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --dirichlet_alpha 0.1 --dataset cifar10 --wandb --task cifar_resnet_1000_dis
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --dirichlet_alpha 0.5 --dataset cifar10 --wandb --task cifar_resnet_1000_dis
python main.py --config conf_1000 --model resnet18  --defense fedavg --attack flinvert --inject_params --dirichlet_alpha 1.0 --dataset cifar10 --wandb --task cifar_resnet_1000_dis


