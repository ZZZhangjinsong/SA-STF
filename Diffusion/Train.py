from typing import Dict
import torch
import numpy as np
from Diffusion.diffusion_res import ResidualDiffusion
from Diffusion.Model import UNet
import data as Data
from osgeo import gdal
import os
from torchvision.utils import make_grid
import math
import random
import torch

def normalization(x):
    average = np.max(x) - np.min(x)
    return (x - np.min(x)) / average


def read_image(image_path, max_data):
    img = gdal.Open(image_path)
    b = np.array(img.ReadAsArray())
    b = b/max_data
    return 2*b-1

def save(arr, save_path, h, l):
    driver = gdal.GetDriverByName("GTiff")
    datasetnew = driver.Create(save_path, l, h, arr.shape[0], gdal.GDT_Float32)
    for i in range(arr.shape[0]):
        band = datasetnew.GetRasterBand(i+1)
        band.WriteArray(arr[i, :, :])
    datasetnew.FlushCache()  # Write to disk.必须有清除缓存

def set_device(x, device):
    if isinstance(x, dict):  # 判断x是否是一个字典
        for key, item in x.items():
            if item is not None:
                x[key] = item.to(device)
    return x

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def train(modelConfig: Dict):
    #设置随机种子
    set_seed(seed=42)
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    # data
    train_set = Data.create_train_dataset(modelConfig)
    train_loader = Data.create_train_dataloader(train_set, modelConfig)
    print('Initial Dataset Finished')
    # model setup
    net_model = UNet(in_channel=modelConfig["in_channel"], out_channel=modelConfig["out_channel"], inner_channel=modelConfig["inner_channel"],
                     norm_groups=modelConfig["norm_groups"], channel_mults=modelConfig["channel_mults"], dropout=modelConfig["dropout"])
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.Adam(net_model.parameters(), lr=modelConfig["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120,160], gamma=0.5)
    trainer = ResidualDiffusion(model_x0=net_model,
                                total_epoch=modelConfig["epoch"],
                                timesteps=modelConfig["T"],
                                sampling_steps=modelConfig["sampling_steps"],
                                ddim_sampling_eta=modelConfig["ddim_sampling_eta"],
                                device=device).to(device)
    print('Initial Model Finished')
    # start training
    for e in range(modelConfig["epoch"]):
        batches = len(train_loader)
        for _, train_data in enumerate(train_loader):
            # train
            optimizer.zero_grad()
            x = set_device(train_data, device)
            loss, loss_feature = trainer(x, e)
            print("total epoch:{}  current epoch:{}  current batch:{} totoal batch:{} loss={} loss_lr_feature={}".format(
                                modelConfig["epoch"], e, _ + 1, batches, loss, loss_feature))
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (e + 1) % 20 == 0 :
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_SA_STF_' + str(e + 1) + ".pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channel=modelConfig["in_channel"], out_channel=modelConfig["out_channel"], inner_channel=modelConfig["inner_channel"],
                     norm_groups=modelConfig["norm_groups"], channel_mults=modelConfig["channel_mults"], dropout=modelConfig["dropout"])
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = ResidualDiffusion(model_x0=model,
                                    total_epoch=modelConfig["epoch"],
                                    timesteps=modelConfig["T"],
                                    sampling_steps=modelConfig["sampling_steps"],
                                    ddim_sampling_eta=modelConfig["ddim_sampling_eta"],
                                    device=device).to(device)
        # Sampled from standard normal distribution
        img_LR = read_image(modelConfig["test_lr_path"], modelConfig["max_data"])
        img_ref0 = read_image(modelConfig["test_ref0_path"], modelConfig["max_data"])
        img_refsr0 = read_image(modelConfig["test_refsr0_path"], modelConfig["max_data"])
        img_ref1 = read_image(modelConfig["test_ref1_path"], modelConfig["max_data"])
        img_refsr1 = read_image(modelConfig["test_refsr1_path"], modelConfig["max_data"])

        images = np.empty([5, img_LR.shape[0], img_LR.shape[1], img_LR.shape[2]], dtype=float)
        images[0] = img_LR
        images[1] = img_ref0
        images[2] = img_ref1
        images[3] = img_refsr0
        images[4] = img_refsr1
        output_image = np.zeros(images[1].shape)
        IMAGE_SIZE = [images[0].shape[1], images[0].shape[2]]
        PATCH_SIZE = 256
        PATCH_STRIDE = PATCH_SIZE // 2
        end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
        end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
        h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
        w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]

        print(w_index_list)
        if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
            h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
        if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
            w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

        k = 0
        n = len(h_index_list) * len(w_index_list)
        for i in range(len(h_index_list)):
            for j in range(len(w_index_list)):
                h_start = h_index_list[i]
                w_start = w_index_list[j]

                img_LR_patch = images[0][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                img_ref0_patch = images[1][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                img_ref1_patch = images[2][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                img_refsr0_patch = images[3][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                img_refsr1_patch = images[4][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                img_LR_patch = torch.tensor(img_LR_patch / 1.0).unsqueeze(0).to(device).float()
                img_ref0_patch = torch.tensor(img_ref0_patch / 1.0).unsqueeze(0).to(device).float()
                img_refsr0_patch = torch.tensor(img_refsr0_patch / 1.0).unsqueeze(0).to(device).float()
                img_ref1_patch = torch.tensor(img_ref1_patch / 1.0).unsqueeze(0).to(device).float()
                img_refsr1_patch = torch.tensor(img_refsr1_patch / 1.0).unsqueeze(0).to(device).float()
                result_patch = sampler.sample(img_LR_patch, img_ref0_patch,
                                                                                  img_refsr0_patch, img_ref1_patch,
                                                                                  img_refsr1_patch)
                result_patch = result_patch.squeeze()
                h_end = h_start + PATCH_SIZE
                w_end = w_start + PATCH_SIZE
                cur_h_start = 0
                cur_h_end = PATCH_SIZE
                cur_w_start = 0
                cur_w_end = PATCH_SIZE

                if i != 0:
                    h_start = h_start + PATCH_SIZE // 4
                    cur_h_start = PATCH_SIZE // 4

                if i != len(h_index_list) - 1:
                    h_end = h_end - PATCH_SIZE // 4
                    cur_h_end = cur_h_end - PATCH_SIZE // 4

                if j != 0:
                    w_start = w_start + PATCH_SIZE // 4
                    cur_w_start = PATCH_SIZE // 4

                if j != len(w_index_list) - 1:
                    w_end = w_end - PATCH_SIZE // 4
                    cur_w_end = cur_w_end - PATCH_SIZE // 4

                output_image[:, h_start: h_end, w_start: w_end] = result_patch[:, cur_h_start: cur_h_end,
                                                                  cur_w_start: cur_w_end].cpu().detach().numpy()
                print("total number:{}  current number:{}".format(n, k + 1))
                k = k + 1
        real_output = (output_image + 1) * 0.5
        save_dir = modelConfig["sampled_dir"] + modelConfig["sampledImgName"]
        save(real_output, save_dir, real_output.shape[1], real_output.shape[2])
