from Diffusion.Train import train, eval

state = "train"
test_load_weight = None

state = "eval"
test_load_weight = "model_zoo/download/DiffusionWeight.pt"

def main(model_config = None):
    modelConfig = {
        "state": state,
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./output/DDPM",
        "test_load_weight": test_load_weight,
        "sampled_dir": "./output/DDPM/SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    from pathlib import Path
    Path(modelConfig["save_weight_dir"]).mkdir(exist_ok=True)
    Path(modelConfig["sampled_dir"]).mkdir(exist_ok=True)
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
