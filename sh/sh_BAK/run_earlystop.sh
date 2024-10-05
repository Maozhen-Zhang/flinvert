
# debug

python main.py --config conf_pretrain_earlystop --defense fedavg --attack dba --wandb
python main.py --config conf_pretrain_earlystop --defense fedavg --attack neurotoxin --wandb
python main.py --config conf_pretrain_earlystop --defense fedavg --attack f3ba --wandb
#python main.py --config conf_pretrain_earlystop --defense fedavg --attack iba --wandb
python main.py --config conf_pretrain_earlystop --defense fedavg --attack cerp --wandb
python main.py --config conf_pretrain_earlystop --defense fedavg --attack flinvert --inject_params --wandb

python main.py --config conf_pretrain_earlystop --defense deepsight --attack dba --wandb
python main.py --config conf_pretrain_earlystop --defense deepsight --attack neurotoxin --wandb
python main.py --config conf_pretrain_earlystop --defense deepsight --attack f3ba --wandb
#python main.py --config conf_pretrain_earlystop --defense deepsight --attack iba --wandb
python main.py --config conf_pretrain_earlystop --defense deepsight --attack cerp --wandb
python main.py --config conf_pretrain_earlystop --defense deepsight --attack flinvert --inject_params --wandb

python main.py --config conf_pretrain_earlystop --defense foolsgold --attack dba --wandb
python main.py --config conf_pretrain_earlystop --defense foolsgold --attack neurotoxin --wandb
python main.py --config conf_pretrain_earlystop --defense foolsgold --attack f3ba --wandb
#python main.py --config conf_pretrain_earlystop --defense foolsgold --attack iba --wandb
python main.py --config conf_pretrain_earlystop --defense foolsgold --attack cerp --wandb
python main.py --config conf_pretrain_earlystop --defense foolsgold --attack flinvert --inject_params --wandb

python main.py --config conf_pretrain_earlystop --defense rlr --attack dba --wandb
python main.py --config conf_pretrain_earlystop --defense rlr --attack neurotoxin --wandb
python main.py --config conf_pretrain_earlystop --defense rlr --attack f3ba --wandb
#python main.py --config conf_pretrain_earlystop --defense rlr --attack iba --wandb
python main.py --config conf_pretrain_earlystop --defense rlr --attack cerp --wandb
python main.py --config conf_pretrain_earlystop --defense rlr --attack flinvert --inject_params --wandb

python main.py --config conf_pretrain_earlystop --defense mkrum --attack dba --wandb
python main.py --config conf_pretrain_earlystop --defense mkrum --attack neurotoxin --wandb
python main.py --config conf_pretrain_earlystop --defense mkrum --attack f3ba --wandb
#python main.py --config conf_pretrain_earlystop --defense mkrum --attack iba --wandb
python main.py --config conf_pretrain_earlystop --defense mkrum --attack cerp --wandb
python main.py --config conf_pretrain_earlystop --defense mkrum --attack flinvert --inject_params --wandb
