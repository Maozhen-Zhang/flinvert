from Terminal.client import Client
from utils.function_normalization import boundOfTransforms


class MalClient(Client):
    def __init__(self,  cfg, ID, train_dataset, test_dataset, identity="None"):
        super(MalClient, self).__init__( cfg, ID, train_dataset, test_dataset, identity="None")

        if cfg.normalize:
            cfg.upper, cfg.lower = boundOfTransforms(cfg)
            cfg.upper, cfg.lower = cfg.upper.to(cfg.device), cfg.lower.to(cfg.device)
        else:
            cfg.upper, cfg.lower = 1, 0

        self.unet = None
        if self.unet == None and cfg.attack == "iba":
            self.unet = self.getUnet(cfg)



    def local_train(self, cfg):
        current_epoch = self.current_epoch
        cfg = self.cfg
        mal_id = cfg.mal_id
        ID = self.ID


        # cfg.local_epoch_mal = cfg.local_epoch
        # cfg.lr_poison = cfg.lr

        # cfg.local_epoch_mal = 5
        # cfg.lr_poison = cfg.lr * 0.2
        #  cfg.lr_poison = cfg.lr * 0.2
        # TODO: 1102 modify to run experiments for attack early stop
        cfg.local_epoch_mal = 2
        cfg.lr_poison = cfg.lr





        cfg.decay_poison = cfg.weight_decay
        cfg.momentum_poison = cfg.momentum

        if ID in mal_id:
            if current_epoch in range(cfg.poison_epoch[0], cfg.poison_epoch[1]):
                print(f"|---Client {ID} train (Poisoned Process)")
                self.malicious_train(self.cfg)
            else:
                # self.benign_train(self.cfg)
                super().local_train(self.cfg)

    def malicious_train(self, cfg):
        raise NotImplementedError("Subclasses should implement this method")


    def getUnet(self, cfg, attack_model=None):
        if cfg.dataset == 'cifar10':
            from Models.unet import UNet
            atkmodel = UNet(3).to(cfg.device)
            # Copy of attack model
            tgtmodel = UNet(3).to(cfg.device)
        elif cfg.dataset == 'mnist':
            from Models.autoencoders import MNISTAutoencoder as Autoencoder
            atkmodel = Autoencoder().to(cfg.device)
            # Copy of attack model
            tgtmodel = Autoencoder().to(cfg.device)

        elif cfg.dataset == 'tiny-imagenet' or cfg.datasetdataset == 'tiny-imagenet32' or cfg.datasetdataset == 'gtsrb':
            if attack_model is None:
                from Models.autoencoders import Autoencoder
                atkmodel = Autoencoder().to(cfg.device)
                tgtmodel = Autoencoder().to(cfg.device)
            elif cfg.attack_model == 'unet':
                from Models.unet import UNet
                atkmodel = UNet(3).to(cfg.device)
                tgtmodel = UNet(3).to(cfg.device)
        else:
            raise Exception(f'Invalid atk model {cfg.dataset}')

        return atkmodel