# # trigger epoch 5
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000


python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack dba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack dba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack dba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack dba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack dba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000

python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack cerp --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack cerp --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack cerp --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack cerp --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack cerp --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000

python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack neurotoxin --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack neurotoxin --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack neurotoxin --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack neurotoxin --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack neurotoxin --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000

python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack f3ba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack f3ba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack f3ba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack f3ba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack f3ba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000

#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
#python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack flinvert --inject_params --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000

