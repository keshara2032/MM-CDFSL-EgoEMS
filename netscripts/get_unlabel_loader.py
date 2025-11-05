from datamodule.unlabel_loader.epic_unlabel_loader import EPICUnlabelLoader
from datamodule.unlabel_loader.meccano_unlabel_loader import MECCANOUnlabelLoader
from datamodule.unlabel_loader.wear_unlabel_loader import WEARUnlabelLoader
# from datamodule.unlabel_loader.egoems_unlabel_loader import EGOEMSUnlabelLoader

def get_unlabel_loader(cfg):
    if cfg.target_dataset.lower() == "epic":
        unlabel_loader = EPICUnlabelLoader(cfg)
    elif cfg.target_dataset.lower() == "meccano":
        unlabel_loader = MECCANOUnlabelLoader(cfg)
    elif cfg.target_dataset.lower() == "wear":
        unlabel_loader = WEARUnlabelLoader(cfg)
    elif cfg.target_dataset.lower() == "egoems":
        raise NotImplementedError("EgoEMS unlabel loader is not implemented yet.")
    else:
        raise Exception(f"{cfg.target_dataset} is not supported!")
    return unlabel_loader
