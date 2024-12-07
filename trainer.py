import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print(cfg)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()