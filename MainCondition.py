from DiffusionFreeGuidence.TrainCondition import train, eval

state = "train"
test_load_weight = None

state = "eval"
test_load_weight = "model_zoo/download/DiffusionConditionWeight.pt"

def main(model_config=None):
    modelConfig = {
        "state": state,
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./output/DDPMCondition/",
        "training_load_weight": None,
        "test_load_weight": test_load_weight,
        "sampled_dir": "./output/DDPMCondition/SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    from pathlib import Path
    Path(modelConfig["save_dir"]).mkdir(exist_ok=True)
    Path(modelConfig["sampled_dir"]).mkdir(exist_ok=True)
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
