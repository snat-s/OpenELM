"""
This module gives an example of how to run the main ELM class.

It uses the hydra library to load the config from the config dataclasses in
configs.py.

This config file demonstrates an example of running ELM with the Sodarace
environment, a 2D physics-based environment in which robots specified by
Python dictionaries are evolved over.

"""
import hydra
from omegaconf import OmegaConf

from openelm import ELM


@hydra.main(
    config_name="elmconfig",
    version_base="1.2",
)
def main(config):
    print(config)
    config['model']['seed'] = 42
    config['model']['batch_size'] = 4
    config['env']['batch_size'] = 4
    print(config)
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)
    elm = ELM(config)
    print("Best Individual: ", elm.run(init_steps=config.qd.init_steps,
                                       total_steps=config.qd.total_steps))


if __name__ == "__main__":
    main()
