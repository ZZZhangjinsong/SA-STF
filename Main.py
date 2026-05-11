from Diffusion.Train import train, eval
import data as Data


def main(model_config = None):
    modelConfig = {
        "state": "train",# train or eval
        "epoch": 200,
        "batch_size": 4,
        "T": 100,
        "sampling_steps": 50,
        "ddim_sampling_eta": 0.,
        "in_channel": 6,
        "out_channel": 6,
        "inner_channel": 64,
        "channel_mults": [1, 2, 4, 8, 8],
        "norm_groups": 32,
        "dropout": 0,
        "lr": 1e-4,
        "multiplier": 2.,
        "h": 256,
        "l": 256,
        "max_data": 10000,
        "train_lr_path": "",
        "train_hr_path": "",
        "train_ref0_path": "",
        "train_refsr0_path": "",
        "train_ref1_path": "",
        "train_refsr1_path": "",
        "test_hr_path": "",
        "test_lr_path": "",
        "test_ref0_path": "",
        "test_refsr0_path": "",
        "test_ref1_path": "",
        "test_refsr1_path": "",
        "use_shuffle": True,
        "num_workers": 4,
        "grad_clip": 1.,
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "",
        "sampled_dir": "./SampledImgs/",
        "sampledImgName": "",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
