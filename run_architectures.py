import logging
import pathlib
import requests
import time
import json
from collections import Counter

from openelm.environments import EvoPromptingEnv
from openelm.mutation_model import DiffModel, MutationModel, PromptModel
from openelm.configs import ArchitectureEnvConfig
from openelm.utils.code_eval import pass_at_k
from openelm.codegen.codegen_utilities import set_seed

import hydra
from omegaconf import OmegaConf

class EvoPrompting:
    def __init__(self, config: ArchitectureEnvConfig) -> None:
        """
            Evaluate EvoPrompting
        """
        self.config: ArchitectureEnvConfig = config
    def run(self):
        """
            Create solutions using EvoPrompting.
        """
        print("Running architectures")
        
@hydra.main(
    config_name="p3config",
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
    evoprompting = EvoPrompting(config)
    print("Best Individual: ", evoprompting.run())

if __name__ == "__main__":
    main()