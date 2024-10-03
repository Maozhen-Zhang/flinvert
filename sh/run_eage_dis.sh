#python main.py --config conf_pretrain --defense fedavg --attack flinvert --dirichlet_alpha 0.001 --inject_params -ie 1900,2100 --wandb --project fl-att-client-1000-result-compare --name ie-2100

#python main.py --config conf_pretrain --defense fedavg --attack flinvert --dirichlet_alpha 0.01 --inject_params -ie 1900,2100 --wandb --project fl-att-client-1000-result-compare --name ie-2100
python main.py --config conf_pretrain --defense fedavg --attack flinvert --dirichlet_alpha 0.01 --inject_params -ie 1900,2100 --wandb --project fl-att-client-1000-result-compare-dis --name ie-2100
python main.py --config conf_pretrain --defense fedavg --attack flinvert --dirichlet_alpha 0.05 --inject_params -ie 1900,2100 --wandb --project fl-att-client-1000-result-compare-dis --name ie-2100
python main.py --config conf_pretrain --defense fedavg --attack flinvert --dirichlet_alpha 0.1 --inject_params -ie 1900,2100 --wandb --project fl-att-client-1000-result-compare-dis --name ie-2100
python main.py --config conf_pretrain --defense fedavg --attack flinvert --dirichlet_alpha 0.3 --inject_params -ie 1900,2100 --wandb --project fl-att-client-1000-result-compare-dis--name ie-2100
python main.py --config conf_pretrain --defense fedavg --attack flinvert --dirichlet_alpha 0.5 --inject_params -ie 1900,2100 --wandb --project fl-att-client-1000-result-compare-dis --name ie-2100
python main.py --config conf_pretrain --defense fedavg --attack flinvert --dirichlet_alpha 1.0 --inject_params -ie 1900,2100 --wandb --project fl-att-client-1000-result-compare-dis --name ie-2100
