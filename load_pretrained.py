import torch

def load_kaggle_pretrained(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Load only compatible layers
    model.temporal[0].weight.data = ckpt["spatial.0.weight"].data.clone()
    model.temporal[1].weight.data = ckpt["spatial.1.weight"].data.clone()
    model.temporal[1].bias.data   = ckpt["spatial.1.bias"].data.clone()

    print("Loaded Kaggle pretrained spatial weights")
