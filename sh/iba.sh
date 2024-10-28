
#python main.py --config conf_earlystop --model resnet18 --n_client 1000 --defense fedavg --attack iba --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop

python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack iba --dataset cifar10 --wandb --task cifar_resnet_iba
python main.py --config conf_earlystop --model resnet18 --n_client 1000 --defense fedavg --attack iba --dataset cifar10 --wandb --task cifar_resnet_1000_earlystop


python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack iba --dataset cifar10 --wandb --task cifar_resnet_iba
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack iba --dataset cifar10 --wandb --task cifar_resnet_iba
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack iba --dataset cifar10 --wandb --task cifar_resnet_iba
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack iba --dataset cifar10 --wandb --task cifar_resnet_iba

python main.py --config conf_vgg11_1000 --model vgg11 --n_client 1000 --defense fedavg --attack iba --dataset cifar10 --wandb --task cifar_vgg_1000
python main.py --config conf_vgg11_1000 --model vgg11 --n_client 1000 --defense deepsight --attack iba --dataset cifar10 --wandb --task cifar_vgg_1000
python main.py --config conf_vgg11_1000 --model vgg11 --n_client 1000 --defense foolsgold --attack iba --dataset cifar10 --wandb --task cifar_vgg_1000
python main.py --config conf_vgg11_1000 --model vgg11 --n_client 1000 --defense mkrum --attack iba --dataset cifar10 --wandb --task cifar_vgg_1000
python main.py --config conf_vgg11_1000 --model vgg11 --n_client 1000 --defense rlr --attack iba --dataset cifar10 --wandb --task cifar_vgg_1000

python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense deepsight --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense foolsgold --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense mkrum --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense rlr --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20

python main.py --config conf_20 --model vgg11 --n_client 20 --defense fedavg --attack iba --dataset cifar10 --wandb --task cifar_vgg_20
python main.py --config conf_20 --model vgg11 --n_client 20 --defense deepsight --attack iba --dataset cifar10 --wandb --task cifar_vgg_20
python main.py --config conf_20 --model vgg11 --n_client 20 --defense foolsgold --attack iba --dataset cifar10 --wandb --task cifar_vgg_20
python main.py --config conf_20 --model vgg11 --n_client 20 --defense mkrum --attack iba --dataset cifar10 --wandb --task cifar_vgg_20
python main.py --config conf_20 --model vgg11 --n_client 20 --defense rlr --attack iba --dataset cifar10 --wandb --task cifar_vgg_20


python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense deepsight --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense foolsgold --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense mkrum --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense rlr --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_1000

python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense deepsight --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense foolsgold --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense mkrum --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense rlr --attack iba --dataset tiny-imagenet --wandb --task cifar_resnet_imagenet_20
