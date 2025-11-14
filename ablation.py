import torch

from models.lsseg import LSSeg
from models.ab import a, b, c, d, e, f, g
from train import *


if __name__ == '__main__':

    loss = tracing_loss
    device = torch.device('cuda')    
    # Load configuration
    config = load_config('config.yaml')

    models = [a, b, c, d, e, f, g, LSSeg]
    for model_name in models:
        desc = os.path.join('ab', model_name.__name__)
        print(desc)

        five_fold_validation(model_name, loss, device, config, desc)

    