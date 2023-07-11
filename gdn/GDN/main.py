# -*- coding: utf-8 -*-
import pickle as pkl
from typing import List, Any

import os
from pathlib import Path
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from GDN.util.env import get_device, set_device
from GDN.util.preprocess import build_loc_net, construct_data
from GDN.util.net_struct import get_feature_map, get_fc_graph_struc

from GDN.datasets.TimeDataset import TimeDataset

from GDN.models.GDN import GDNModule

from GDN.train import train
from GDN.test import test

import GDN.dataloader_fix


def GDNtrain(train_config: dict, env_config: dict) -> None:
    feature_map = get_feature_map(env_config["dataset"])
    fc_struc = get_fc_graph_struc(env_config["dataset"])

    set_device(env_config["device"]
               if "device" in env_config else
               "cpu")
    device = get_device()

    fc_edge_index = build_loc_net(fc_struc, list(env_config["dataset"].columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

    train_ts = env_config["dataset"]
    train_dataset_indata = construct_data(train_ts, feature_map, labels=0)

    cfg = {
        'slide_win': train_config['slide_win'],
        'slide_stride': train_config['slide_stride'],
    }

    train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
    train_dataloader, val_dataloader = get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                   val_ratio=train_config['val_ratio'])
    full_train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch'],
                                       shuffle=False, num_workers=0)

    edge_index_sets = []
    edge_index_sets.append(fc_edge_index)

    model = create_gdn_model(train_config, edge_index_sets,
                             feature_map, device)

    model_save_path = get_save_path(env_config["save_model_path"])[0]

    train_log = train(model, model_save_path,
                      config=train_config,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      feature_map=feature_map,
                      test_dataloader=None,
                      test_dataset=None,
                      train_dataset=train_dataset,
                      dataset_name=env_config['dataset']
                      )

    _, train_result = test(model, full_train_dataloader)
    save_result_output(train_result, env_config)

    save_config_element(env_config["save_model_path"],
                        train_config, feature_map, fc_edge_index)


def GDNtest(env_config: dict) -> None:
    elements = load_config_element()
    train_config, feature_map, fc_edge_index = elements[0], elements[1], elements[2]

    set_device(env_config["device"]
               if "device" in env_config else
               "cpu")
    device = get_device()

    edge_index_sets = []
    edge_index_sets.append(fc_edge_index)

    test_ts = env_config["dataset"]
    test_dataset_indata = construct_data(test_ts, feature_map,
                                         labels=test["is_anomaly"].tolist())

    cfg = {
        'slide_win': train_config['slide_win'],
        'slide_stride': train_config['slide_stride'],
    }

    test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

    test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                 shuffle=False, num_workers=0)

    model_load_path = get_save_path(env_config["load_model_path"])[0]

    model = create_gdn_model(train_config, edge_index_sets,
                             feature_map, device)

    model.load_state_dict(torch.load(model_load_path))
    best_model = model.to(device)

    _, test_result = test(best_model, test_dataloader)
    save_result_output(test_result, env_config["dataOutput"])


def create_gdn_model(train_config,
                     edge_index_sets, feature_map,
                     device) -> GDN:
    return GDNModule(edge_index_sets, len(feature_map),
                     dim=train_config['dim'],
                     input_dim=train_config['slide_win'],
                     out_layer_num=train_config['out_layer_num'],
                     out_layer_inter_dim=train_config['out_layer_inter_dim'],
                     topk=train_config['topk']
                     ).to(device)


def get_loaders(train_dataset, seed, batch, val_ratio=0.1):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch,
                                  shuffle=True)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

    return train_dataloader, val_dataloader


def get_save_path(model_path: str):
    # TODO change to TimeEval format of given locations
    if len(model_path) > 0:
        base_dir = os.path.dirname(model_path)
    else:
        base_dir = "results"

    now = datetime.now()
    datestr = now.strftime('%m|%d-%H:%M:%S')

    paths = [
        f'{base_dir}/best_{datestr}.pt',
        f'{base_dir}/results.tmp',
        f'{base_dir}/train_config.pkl',
        f'{base_dir}/feature_map.pkl',
        f'{base_dir}/fc_edge_index.pkl'
    ]

    for path in paths:
        dirname = os.path.dirname(path)
        Path(dirname).mkdir(parents=True, exist_ok=True)

    return paths


def save_config_element(model_path, train_config, feature_map, fc_edge_index) -> None:
    paths = get_save_path(model_path)
    for p, e in zip(paths[-3:],
                    [train_config, feature_map, fc_edge_index]):
        with open(p, 'wb') as file:
            pkl.dump(e, file, protocol=pkl.HIGHEST_PROTOCOL)


def load_config_element() -> List[Any]:
    paths = get_save_path()
    elements = []
    for p in paths[-3:]:
        if not Path(p).exists():
            raise FileNotFoundError("Base element not found in required path."
                                    "Run training first", p)
        with open(p, 'rb') as file:
            elements.append(pkl.load(file))
    return elements


def save_result_output(result, env_config) -> None:
    path = get_save_path(env_config["save_model_path"])[1].replace("results.tmp", env_config["dataOutput"])
    np_result = np.array(result)
    np.savetxt(path, np_result, delimiter=",")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-batch', help='batch size', type=int, default=128)
#     parser.add_argument('-epoch', help='train epoch', type=int, default=100)
#     parser.add_argument('-slide_win', help='slide_win', type=int, default=15)
#     parser.add_argument('-dim', help='dimension', type=int, default=64)
#     parser.add_argument('-slide_stride', help='slide_stride', type=int, default=5)
#     parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
#     parser.add_argument('-dataset', help='wadi / swat', type=str, default='wadi')
#     parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
#     parser.add_argument('-random_seed', help='random seed', type=int, default=0)
#     parser.add_argument('-comment', help='experiment comment', type=str, default='')
#     parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
#     parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=256)
#     parser.add_argument('-decay', help='decay', type=float, default=0)
#     parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
#     parser.add_argument('-topk', help='topk num', type=int, default=20)
#     parser.add_argument('-report', help='best / val', type=str, default='best')
#     parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
#
#     args = parser.parse_args()
#
#     random.seed(args.random_seed)
#     np.random.seed(args.random_seed)
#     torch.manual_seed(args.random_seed)
#     torch.cuda.manual_seed(args.random_seed)
#     torch.cuda.manual_seed_all(args.random_seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     os.environ['PYTHONHASHSEED'] = str(args.random_seed)
#
#     train_config = {
#         'batch': args.batch,
#         'epoch': args.epoch,
#         'slide_win': args.slide_win,
#         'dim': args.dim,
#         'slide_stride': args.slide_stride,
#         'comment': args.comment,
#         'seed': args.random_seed,
#         'out_layer_num': args.out_layer_num,
#         'out_layer_inter_dim': args.out_layer_inter_dim,
#         'decay': args.decay,
#         'val_ratio': args.val_ratio,
#         'topk': args.topk,
#     }
#
#     env_config = {
#         'save_path': args.save_path_pattern,
#         'dataset': args.dataset,
#         'report': args.report,
#         'device': args.device,
#         'load_model_path': args.load_model_path
#     }
#
#     main = GDNMain(train_config, env_config, debug=False)
#     main.run()
