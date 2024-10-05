
# vgg11 200 client
python main.py --config conf_200 --model vgg11 --n_client 200 --defense fedavg --attack dba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense fedavg --attack neurotoxin --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense fedavg --attack cerp --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense fedavg --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_200 --model vgg11 --n_client 200 --defense fedavg --attack iba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense fedavg --attack flinvert --inject_params --dataset cifar10 --wandb

python main.py --config conf_200 --model vgg11 --n_client 200 --defense deepsight --attack dba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense deepsight --attack neurotoxin --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense deepsight --attack cerp --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense deepsight --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_200 --model vgg11 --n_client 200 --defense deepsight --attack iba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense deepsight --attack flinvert --inject_params --dataset cifar10 --wandb


python main.py --config conf_200 --model vgg11 --n_client 200 --defense foolsgold --attack dba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense foolsgold --attack neurotoxin --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense foolsgold --attack cerp --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense foolsgold --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_200 --model vgg11 --n_client 200 --defense foolsgold --attack iba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense foolsgold --attack flinvert --inject_params --dataset cifar10 --wandb

python main.py --config conf_200 --model vgg11 --n_client 200 --defense mkrum --attack dba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense mkrum --attack neurotoxin --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense mkrum --attack cerp --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense mkrum --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_200 --model vgg11 --n_client 200 --defense mkrum --attack iba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense mkrum --attack flinvert --inject_params --dataset cifar10 --wandb

python main.py --config conf_200 --model vgg11 --n_client 200 --defense rlr --attack dba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense rlr --attack neurotoxin --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense rlr --attack cerp --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense rlr --attack f3ba --dataset cifar10 --wandb
#python main.py --config conf_200 --model vgg11 --n_client 200 --defense rlr --attack iba --dataset cifar10 --wandb
python main.py --config conf_200 --model vgg11 --n_client 200 --defense rlr --attack flinvert --inject_params --dataset cifar10 --wandb
