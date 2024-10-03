import os


def mk_folder(cfg):
    mkdir_path = './visual'
    if not os.path.exists(mkdir_path):
        os.makedirs(mkdir_path)
        print(f"\n|---Folder '{mkdir_path}' created. \n")
    else:
        print(f"\n|---Folder '{mkdir_path}' already exists.\n")

    mkdir_path = f'./visual/{cfg.attack}'
    if not os.path.exists(mkdir_path):
        os.makedirs(mkdir_path)
        print(f"\n|---Folder '{mkdir_path}' created. \n")
    else:
        print(f"\n|---Folder '{mkdir_path}' already exists.\n")

    mkdir_path = f'./visual/{cfg.attack}/triggers'
    if not os.path.exists(mkdir_path):
        os.makedirs(mkdir_path)
        print(f"\n|---Folder '{mkdir_path}' created. \n")
    else:
        print(f"\n|---Folder '{mkdir_path}' already exists.\n")


    #### result
    mkdir_path = './results'
    if not os.path.exists(mkdir_path):
        os.makedirs(mkdir_path)
        print(f"\n|---Folder '{mkdir_path}' created. \n")
    else:
        print(f"\n|---Folder '{mkdir_path}' already exists.\n")


    mkdir_path = f'./results/{cfg.attack}'
    if not os.path.exists(mkdir_path):
        os.makedirs(mkdir_path)
        print(f"\n|---Folder '{mkdir_path}' created. \n")
    else:
        print(f"\n|---Folder '{mkdir_path}' already exists.\n")


    mkdir_path = f'./checkpoints-new/'
    if not os.path.exists(mkdir_path):
        os.makedirs(mkdir_path)
        print(f"\n|---Folder '{mkdir_path}' created. \n")
    else:
        print(f"\n|---Folder '{mkdir_path}' already exists.\n")

    mkdir_path = f'./checkpoints-new/{cfg.model}-{cfg.dataset}-{cfg.attack}-{cfg.defense}-{cfg.n_client}'
    if not os.path.exists(mkdir_path):
        os.makedirs(mkdir_path)
        print(f"\n|---Folder '{mkdir_path}' created. \n")
    else:
        print(f"\n|---Folder '{mkdir_path}' already exists.\n")
