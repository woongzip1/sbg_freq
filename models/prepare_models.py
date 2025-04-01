""" State model maps here """
import torch
from torchinfo import summary

from box import Box
from models.model import APNet_BWE_Model
from models.seanet import SEANet
from models.discriminators import MultiBandSTFTDiscriminator, SSDiscriminatorBlock

import sys
sys.path.append("..")
from utils import count_model_params

MODEL_MAP = {
    'generator': APNet_BWE_Model,
    'seanet': SEANet,
    }
DISCRIMINATOR_KWARG_MAP = {
        "MultiBandSTFTDiscriminator": lambda cfg: {
            "sd_num": len(cfg.n_fft_list),
            "C": cfg.C,
            "n_fft_list": cfg.n_fft_list,
            "hop_len_list": cfg.hop_len_list,
            "band_split_ratio": cfg.band_split_ratio,
            "sd_mode": "BS",
        },
        "PeriodDiscriminator": lambda cfg: {
            "pd_num": len(cfg.period_list),
            "period_list": cfg.period_list,
            "C_period": cfg.C_period,
        }
    }

def prepare_discriminator(config:Box):
    types = config.discriminator.types
    configs = config.discriminator.configs
    if not types:
        raise ValueError(f"At least one discriminator is required")

    kwargs = {}
    for disc_type in types:
        if disc_type not in DISCRIMINATOR_KWARG_MAP:
            raise ValueError(f"Unsupported discriminator type: {disc_type}")
        kwargs.update(DISCRIMINATOR_KWARG_MAP[disc_type](configs[disc_type]))

    discriminator = SSDiscriminatorBlock(**kwargs)

    # Print information about the loaded model
    print("########################################")
    print("🚀 Discriminator Configurations:")
    for t in types:
        print(f"- {t}:")
        for k, v in configs[t].items():
            print(f"    {k}: {v}")

    p = count_model_params(discriminator)
    print(f"✅ Discriminator Parameters: {p/1_000_000:.2f}M")
    print("########################################")

    return discriminator

def prepare_generator(config:Box, MODEL_MAP):
    import inspect
    gen_type = config.generator.type
    if gen_type not in MODEL_MAP:
        raise ValueError(f"Unsupported generator type: {gen_type}")
    
    GenClass = MODEL_MAP[gen_type]
    sig = inspect.signature(GenClass.__init__)
    if "h" in sig.parameters:
        hparams = config.generator.hparams
        generator = GenClass(hparams)
    else:        
        hparams = config.generator.hparams
        generator = GenClass(**hparams)

    # Logging
    print("########################################")
    print(f"🚀 Instantiating Generator: {gen_type}")
    for k, v in hparams.items():
        print(f"  {k}: {v}")
    print("########################################")

    # Count parameters
    p = count_model_params(generator)
    print(f"✅ Generator Parameters: {p/1_000_000:.2f}M")

    # Dummy forward
    lr = torch.randn(1,1,48000)
    mag_nb = torch.randn(1,513,400)
    pha_nb = torch.randn(1,513,400)
    try:
        if gen_type == "seanet":
            summary(generator, input_data=[lr], depth=2,
                col_names=["input_size", "output_size", "num_params",])
        else:
            summary(generator, input_data=[mag_nb, pha_nb], depth=2,
                col_names=["input_size", "output_size", "num_params",])
        
    except Exception as e:
        print(f"⚠️ torchinfo.summary failed: {e}")
        
        

    return generator

def main():
    from utils import print_config
    from main import load_config
    
    config_path = "configs/config_template.yaml" # 8, 21
    config = load_config(config_path)
    disc = prepare_discriminator(config)
    gen = prepare_generator(config, MODEL_MAP)
    
if __name__ == "__main__":
    ### python -m models.prepare_models
    main()
    