
# debug

#python main.py --config conf_imagenet --defense fedavg --attack noatt --wandb --project flinvert-imagenet

#python main.py --config conf_imagenet --defense fedavg --attack cerp --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet --defense fedavg --attack dba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet --defense fedavg --attack iba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet --defense fedavg --attack neurotoxin --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet --defense fedavg --attack f3ba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet --defense fedavg --attack flinvert --inject_params --wandb --project flinvert-imagenet

#python main.py --config conf_imagenet_1000 --defense fedavg --attack noatt --wandb --project flinvert-imagenet
#
#python main.py --config conf_imagenet_1000 --defense fedavg --attack cerp --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000 --defense fedavg --attack dba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000 --defense fedavg --attack iba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000 --defense fedavg --attack neurotoxin --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000 --defense fedavg --attack f3ba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000 --defense fedavg --attack flinvert --inject_params --wandb --project flinvert-imagenet


#python main.py --config conf_imagenet_1000_earlystop --defense fedavg --attack cerp --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000_earlystop --defense fedavg --attack dba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000_earlystop --defense fedavg --attack iba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000_earlystop --defense fedavg --attack neurotoxin --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000_earlystop --defense fedavg --attack f3ba --wandb --project flinvert-imagenet
#python main.py --config conf_imagenet_1000_earlystop --defense fedavg --attack flinvert --inject_params --wandb --project flinvert-imagenet

python main.py --config conf_imagenet_1000_earlystop --defense fedavg --attack flinvert --inject_params --wandb --project flinvert-mnist
python main.py --config conf_imagenet_1000_earlystop --defense deepsight --attack flinvert --inject_params --wandb --project flinvert-mnist
python main.py --config conf_imagenet_1000_earlystop --defense foolsgold --attack flinvert --inject_params --wandb --project flinvert-mnist
python main.py --config conf_imagenet_1000_earlystop --defense mkrum --attack flinvert --inject_params --wandb --project flinvert-mnist
python main.py --config conf_imagenet_1000_earlystop --defense rlr --attack flinvert --inject_params --wandb --project flinvert-mnist

