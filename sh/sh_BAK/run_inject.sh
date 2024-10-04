
# debug

python main.py --config conf --defense fedavg --attack dba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense fedavg --attack neurotoxin --wandb --project fl-att-client-20-compare
python main.py --config conf --defense fedavg --attack f3ba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense fedavg --attack iba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense fedavg --attack cerp --wandb --project fl-att-client-20-compare
python main.py --config conf --defense fedavg --attack flinvert --inject_params --wandb --project fl-att-client-20-compare

python main.py --config conf --defense deepsight --attack dba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense deepsight --attack neurotoxin --wandb --project fl-att-client-20-compare
python main.py --config conf --defense deepsight --attack f3ba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense deepsight --attack iba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense deepsight --attack cerp --wandb --project fl-att-client-20-compare
python main.py --config conf --defense deepsight --attack flinvert --inject_params --wandb --project fl-att-client-20-compare

python main.py --config conf --defense foolsgold --attack dba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense foolsgold --attack neurotoxin --wandb --project fl-att-client-20-compare
python main.py --config conf --defense foolsgold --attack f3ba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense foolsgold --attack iba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense foolsgold --attack cerp --wandb --project fl-att-client-20-compare
python main.py --config conf --defense foolsgold --attack flinvert --wandb --project fl-att-client-20-compare

python main.py --config conf --defense rlr --attack dba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense rlr --attack neurotoxin --wandb --project fl-att-client-20-compare
python main.py --config conf --defense rlr --attack f3ba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense rlr --attack iba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense rlr --attack cerp --wandb --project fl-att-client-20-compare
python main.py --config conf --defense rlr --attack flinvert --inject_params --wandb --project fl-att-client-20-compare

python main.py --config conf --defense mkrum --attack dba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense mkrum --attack neurotoxin --wandb --project fl-att-client-20-compare
python main.py --config conf --defense mkrum --attack f3ba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense mkrum --attack iba --wandb --project fl-att-client-20-compare
python main.py --config conf --defense mkrum --attack cerp --wandb --project fl-att-client-20-compare
python main.py --config conf --defense mkrum --attack flinvert --inject_params --wandb --project fl-att-client-20-compare


python main.py --config conf --defense fedavg --attack flinvert --threshold 0.0001 --delta 0.0001 --inject_params --wandb --project fl-att-client-20-compare
