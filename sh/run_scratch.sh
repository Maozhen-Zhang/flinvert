python main.py --config conf --defense fedavg --attack dba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense fedavg --attack neurotoxin  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense fedavg --attack f3ba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense fedavg --attack iba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense fedavg --attack cerp  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense fedavg --attack flinvert  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense fedavg --attack flinvert --inject_params -ie 0,200 --wandb --project flinvert-20-result-compare --name ie-200


python main.py --config conf --defense deepsight --attack dba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense deepsight --attack neurotoxin  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense deepsight --attack f3ba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense deepsight --attack iba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense deepsight --attack cerp  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense deepsight --attack flinvert  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense deepsight --attack flinvert --inject_params -ie 0,200 --wandb --project flinvert-20-result-compare --name ie-200


python main.py --config conf --defense foolsgold --attack dba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense foolsgold --attack neurotoxin  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense foolsgold --attack f3ba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense foolsgold --attack iba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense foolsgold --attack cerp  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense foolsgold --attack flinvert  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense foolsgold --attack flinvert --inject_params -ie 0,200 --wandb --project flinvert-20-result-compare --name ie-200


python main.py --config conf --defense rlr --attack dba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense rlr --attack neurotoxin  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense rlr --attack f3ba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense rlr --attack iba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense rlr --attack cerp  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense rlr --attack flinvert  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense rlr --attack flinvert --inject_params -ie 0,200 --wandb --project flinvert-20-result-compare --name ie-200


python main.py --config conf --defense mkrum --attack dba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense mkrum --attack neurotoxin  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense mkrum --attack f3ba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense mkrum --attack iba  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense mkrum --attack cerp  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense mkrum --attack flinvert  --wandb --project flinvert-20-result-compare
python main.py --config conf --defense mkrum --attack flinvert --inject_params -ie 0,200 --wandb --project flinvert-20-result-compare --name ie-200

