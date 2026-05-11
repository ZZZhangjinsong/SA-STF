'''create data and dataloader'''
import logging
from re import split
import torch.utils.data
from typing import Dict


def create_train_dataloader(dataset, modelConfig: Dict):
    '''create dataloader '''
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=modelConfig['batch_size'],
            shuffle=modelConfig['use_shuffle'],#在每次开始对数据重新排序
            num_workers=modelConfig['num_workers'],
            pin_memory=False)

def create_train_dataset(modelConfig: Dict):
    '''create data'''
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(lr_path = modelConfig["train_lr_path"],
                hr_path = modelConfig["train_hr_path"],
                ref0_path = modelConfig["train_ref0_path"],
                refsr0_path = modelConfig["train_refsr0_path"],
                ref1_path=modelConfig["train_ref1_path"],
                refsr1_path=modelConfig["train_refsr1_path"],
                h = modelConfig["h"],
                l = modelConfig["l"],
                split = modelConfig["state"],
                maxdata = modelConfig["max_data"]
                )
    logger = logging.getLogger('base')
    logger.info('Dataset is created.')
    return dataset
