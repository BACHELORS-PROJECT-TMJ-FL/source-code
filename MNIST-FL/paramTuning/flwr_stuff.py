from .typing import Config
import subprocess

import os

def config_to_cli_format(config: dict) -> str:
    all_options = ""
    for key, value in config.items():
        if isinstance(value, str):
            all_options += f"{key}='{value}' "
        elif isinstance(value, bool):
            all_options += f"{key}={str(value).lower()} "
        elif isinstance(value, int) or isinstance(value, float):
            all_options += f"{key}={value} "
        else:
            raise ValueError(f"Unsupported type for key {key}: {type(value)}")
            
    return f"\"{all_options}\""

def run_flower_simulation(config: Config) -> bool:
    """
    Executes a Flower federated learning experiment with the given configuration.
    
    Args:
        config: Dictionary containing experiment configuration parameters
        fed_config: Dictionary containing federation-specific configuration
        
    Returns:
        bool: True if the experiment was executed successfully, 
              False otherwise
    """
    
    run_config = config_to_cli_format(config)
    # command = f"flwr run {run_config} {fed_config}"
    
    num_supernodes = 5 #config["num-clients"]
    command = f"""flower-simulation --app . \
                            --num-supernodes {num_supernodes} \
                            --run-config {run_config}"""
    
    print(f"Executing: {command}")
    try:
        # Execute the command and capture the output
        completed_process = subprocess.run(command, 
                                           shell=True, 
                                           check=True, 
                                           capture_output=False, 
                                           text=True)
    
        print(f"Command output: {completed_process}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the experiment: {e}")
        return False



def run_flower_simulation_async(config: Config) -> subprocess.Popen:
    """
    Executes a Flower federated learning experiment asynchronously without showing outputs.
    
    Args:
        config: Dictionary containing experiment configuration parameters
        
    Returns:
        subprocess.Popen: The process object of the running command
    """
    
    run_config = config_to_cli_format(config)
    num_supernodes = 5 # config["num-clients"]
    command = f"""flower-simulation --app . \
                            --num-supernodes {num_supernodes} \
                            --run-config {run_config}"""
    
    # Redirect output to devnull to suppress it
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=devnull,
            stderr=devnull,
            text=True
        )
    
    return process