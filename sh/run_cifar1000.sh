#python main.py --config conf_mnist --defense fedavg --attack noatt --wandb --project flinvert-mnist
#python main.py --config conf_mnist --defense fedavg --attack flinvert --wandb --project flinvert-mnist
#python main.py --config conf_mnist --defense fedavg --attack flinvert --inject_params --wandb --project flinvert-mnist
#python main.py --config conf_mnist --defense deepsight --attack flinvert --inject_params --wandb --project flinvert-mnist
#python main.py --config conf_mnist --defense foolsgold --attack flinvert --inject_params --wandb --project flinvert-mnist
#python main.py --config conf_mnist --defense mkrum --attack flinvert --inject_params --wandb --project flinvert-mnist
#python main.py --config conf_mnist --defense rlr --attack flinvert --inject_params --wandb --project flinvert-mnist
#
#
#python main.py --config conf_cifar10 --defense fedavg --attack flinvert --inject_params --wandb --project flinvert-mnist
#python main.py --config conf_mnist --defense mkrum --attack flinvert --inject_params
#python main.py --config conf_pretain --defense mkrum --attack flinvert



#python main.py --config conf_cifar10 --defense fedavg --attack noatt --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10 --defense fedavg --attack noatt --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10 --defense fedavg --attack noatt --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10 --defense fedavg --attack noatt --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10 --defense fedavg --attack flinvert --wandb --project flinvert-vgg-earlystop



#python main.py --config conf_cifar10 --defense fedavg --attack noatt --wandb --project flinvert-vgg-earlystop

python main.py --config conf_cifar10_earlystop --defense fedavg --attack flinvert --inject_params --threshold 0.0005 --delta 0.0001 --wandb --project flinvert-vgg-earlystop
python main.py --config conf_cifar10_earlystop --defense fedavg --attack flinvert --inject_params --threshold 0.001 --delta 0.0005 --wandb --project flinvert-vgg-earlystop
python main.py --config conf_cifar10_earlystop --defense fedavg --attack flinvert --inject_params --threshold 0.0001 --delta 0.001 --wandb --project flinvert-vgg-earlystop
python main.py --config conf_cifar10_earlystop --defense fedavg --attack flinvert --inject_params --threshold 0.0005 --delta 0.001 --wandb --project flinvert-vgg-earlystop
python main.py --config conf_cifar10_earlystop --defense fedavg --attack flinvert --inject_params --threshold 0.005 --delta 0.001 --wandb --project flinvert-vgg-earlystop

#python main.py --config conf_cifar10_earlystop --defense fedavg --attack dba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense fedavg --attack neurotoxin --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense fedavg --attack iba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense fedavg --attack f3ba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense fedavg --attack cerp --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense fedavg --attack flinvert --inject_params --wandb --project flinvert-vgg-earlystop

#python main.py --config conf_cifar10_earlystop --defense deepsight --attack dba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense deepsight --attack neurotoxin --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense deepsight --attack iba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense deepsight --attack f3ba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense deepsight --attack cerp --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense deepsight --attack flinvert --inject_params --wandb --project flinvert-vgg-earlystop
#
#python main.py --config conf_cifar10_earlystop --defense foolsgold --attack dba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense foolsgold --attack neurotoxin --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense foolsgold --attack iba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense foolsgold --attack f3ba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense foolsgold --attack cerp --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense foolsgold --attack flinvert --inject_params --wandb --project flinvert-vgg-earlystop
#
#python main.py --config conf_cifar10_earlystop --defense mkrum --attack dba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense mkrum --attack neurotoxin --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense mkrum --attack iba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense mkrum --attack f3ba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense mkrum --attack cerp --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense mkrum --attack flinvert --inject_params --wandb --project flinvert-vgg-earlystop
#
#python main.py --config conf_cifar10_earlystop --defense rlr --attack dba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense rlr --attack neurotoxin --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense rlr --attack iba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense rlr --attack f3ba --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense rlr --attack cerp --wandb --project flinvert-vgg-earlystop
#python main.py --config conf_cifar10_earlystop --defense rlr --attack flinvert --inject_params --wandb --project flinvert-vgg-earlystop
