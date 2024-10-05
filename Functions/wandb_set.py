import wandb


def wandb_setting(cfg):
    wandb_name = cfg.dataset + '-' + cfg.model + '-' +cfg.defense + '-' + cfg.attack + '-' + str(cfg.lr) + '-' + str(cfg.agglr) + '-' + str(
        cfg.n_client) + '-' + str(cfg.sample) + '-' + str(len(cfg.mal_id) / cfg.n_client) + '-' + str(cfg.dirichlet_alpha)
    if cfg.attack == "flinvert":
        wandb_name += '-' + str(cfg.epsilon)
        if cfg.inject_params:
            wandb_name += f'-inject_params-{cfg.inject_params}'
            wandb_name += f'-threshold-{cfg.threshold}'
            wandb_name += f'-delta-{cfg.delta}'
            wandb_name += f'-inject_epoch-{cfg.inject_epoch}'
    wandb_name += '-' + str(cfg.name)
    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.project,
        # group=cfg.dataset + '-' + cfg.model + '-' + cfg.defense + '-' + str(cfg.n_client),
        group=str(cfg.task),

        name=wandb_name,
        # track hyperparameters and run metadata
        tags=[cfg.dataset, cfg.model, cfg.defense, cfg.attack],
        config={
            "settings": cfg.__dict__,
        }) if cfg.wandb else None

