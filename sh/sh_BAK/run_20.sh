python main.py --config conf --defense fedavg --attack flinvert --inject_params -ie 1900,2100 --wandb --project flinvert-20-result-compare --name ie-2100
python main.py --config conf --defense foolsgold --attack flinvert --inject_params -ie 1900,2100 --wandb --project flinvert-20-result-compare --name ie-2100
python main.py --config conf --defense deepsight --attack flinvert --inject_params -ie 1900,2100 --wandb --project flinvert-20-result-compare --name ie-2100
python main.py --config conf --defense mkrum --attack flinvert --inject_params -ie 1900,2100 --wandb --project flinvert-20-result-compare --name ie-2100
python main.py --config conf --defense rlr --attack flinvert --inject_params -ie 1900,2100 --wandb --project flinvert-20-result-compare --name ie-2100
