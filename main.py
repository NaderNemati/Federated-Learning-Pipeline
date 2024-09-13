from pathlib import Path
import pickle
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
import flwr as fl
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    
    # 2. Dataset preparation
    # Dataset partition
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, 
                                                                cfg.batch_size)

    print(len(trainloaders), len(trainloaders[0].dataset))

    # 3. Clients definition
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)


    # 4. Strategy definition
    '''
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))
    '''
    strategy = fl.server.strategy.FedAvg(
                                    fraction_fit=0.6,  # Fraction of clients used for training (3 clients out of 5)
                                    min_fit_clients=3,  # Minimum number of clients required for training
                                    fraction_evaluate=0.6,  # Fraction of clients used for evaluation (2 clients out of 5)
                                    min_evaluate_clients=2,  # Minimum number of clients required for evaluation
                                    min_available_clients=5  # Total number of clients available
    )


    #strategy = instantiate(cfg.strategy,
    #                       on_fit_config_fn=get_on_fit_config(cfg.config_fit),
    #                       get_evaluate_fn=get_evaluate_fn(cfg.num_classes,testloader)
    #                       )


    #strategy = instantiate(cfg.strategy, get_evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))
    #strategy = instantiate(cfg.strategy) if 'strategy' in cfg else fl.server.strategy.FedAvg(
    #                            fraction_fit=0.6,
    #                            min_fit_clients=cfg.num_clients_per_round_fit,
    #                            fraction_evaluate=0.5,
    #                            min_evaluate_clients=cfg.num_clients_per_round_eval,
    #                            min_available_clients=cfg.num_clients,
    #                            )


    # 5. Start simulation
    history = fl.simulation.start_simulation(
        client_fn = client_fn,
        num_clients = cfg.num_clients,
        config = fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy = strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 0.0 },
    )


    # 6. Saving the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'

    results = {'history': history}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol = pickle.HIGHEST_PROTOCOL)






if __name__ == "__main__":
    main()