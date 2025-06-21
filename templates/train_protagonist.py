import gymnasium as gym
import yaml
import numpy as np
import os
import tempfile
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from wandb.integration.sb3 import WandbCallback
import wandb

from WindGym import WindFarmEnv  # Using WindFarmEnv for training
from py_wake.examples.data.hornsrev1 import V80


class WindGymCustomMonitor(BaseCallback):
    """
    A custom callback for logging detailed environment-specific metrics from WindGym to Weights & Biases.
    This version assumes per-turbine TI is available in the `info` dictionary.
    """

    def __init__(self, verbose=0):
        super(WindGymCustomMonitor, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This function is called after each step in the training process.
        It iterates through the `infos` dictionary from each parallel environment
        and logs the data to Weights & Biases.
        """
        infos = self.locals["infos"]

        for env_idx in range(len(infos)):
            info = infos[env_idx]
            log_data = {}

            # 1. Log global environment conditions
            log_data[f"custom_env_{env_idx}/global_wind_speed"] = info.get(
                "Wind speed Global"
            )
            log_data[f"custom_env_{env_idx}/global_wind_dir"] = info.get(
                "Wind direction Global"
            )
            log_data[f"custom_env_{env_idx}/global_ti"] = info.get(
                "Turbulence intensity"
            )

            # 2. Log farm-level power metrics
            agent_power = info.get("Power agent", 0)
            baseline_power = info.get("Power baseline")

            log_data[f"custom_env_{env_idx}/total_agent_power"] = agent_power

            if baseline_power is not None:
                log_data[f"custom_env_{env_idx}/total_baseline_power"] = baseline_power
                if baseline_power > 0:
                    power_gain = agent_power / baseline_power
                    log_data[f"custom_env_{env_idx}/power_gain_vs_baseline"] = (
                        power_gain
                    )

            # 3. Log per-turbine data from the 'info' dictionary
            num_turbines = len(info.get("yaw angles agent", []))
            turbine_tis = info.get("Turbulence intensity at turbines")

            for i in range(num_turbines):
                log_data[f"custom_env_{env_idx}/turbine_{i}/yaw_angle"] = info.get(
                    "yaw angles agent", [np.nan] * num_turbines
                )[i]
                log_data[f"custom_env_{env_idx}/turbine_{i}/power"] = info.get(
                    "Power pr turbine agent", [np.nan] * num_turbines
                )[i]
                log_data[f"custom_env_{env_idx}/turbine_{i}/local_wind_speed"] = (
                    info.get("Wind speed at turbines", [np.nan] * num_turbines)[i]
                )
                if "Wind direction at turbines" in info:
                    log_data[f"custom_env_{env_idx}/turbine_{i}/local_wind_dir"] = (
                        info.get(
                            "Wind direction at turbines", [np.nan] * num_turbines
                        )[i]
                    )
                if turbine_tis is not None:
                    log_data[
                        f"custom_env_{env_idx}/turbine_{i}/turbulence_intensity"
                    ] = turbine_tis[i]

            wandb.log(log_data, step=self.num_timesteps)

        return True


def make_env(temp_yaml_path, turbine_obj, env_init_params, seed=0):
    """
    Factory function for creating a WindFarmEnv instance for SubprocVecEnv.
    """
    def _init():
        # Each subprocess loads the configuration from the temporary YAML file.
        with open(temp_yaml_path, "r") as f:
            config = yaml.safe_load(f)

        farm_params = config["farm"]
        nx = farm_params["nx"]
        ny = farm_params["ny"]
        x_dist = farm_params["xDist"]
        y_dist = farm_params["yDist"]
        D = turbine_obj.diameter()

        x_coords = np.arange(nx) * x_dist * D
        y_coords = np.arange(ny) * y_dist * D
        x_pos, y_pos = np.meshgrid(x_coords, y_coords)
        x_pos, y_pos = x_pos.flatten(), y_pos.flatten()

        env = WindFarmEnv(
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=turbine_obj,
            yaml_path=temp_yaml_path,
            seed=seed,
            **env_init_params,
            Baseline_comp=True, # Always enable for comparison logging
        )
        return env

    return _init


def train_agent(args):
    """
    Main function to set up and run the training process.
    """
    # 1. Load the base YAML configuration from the specified file
    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config_path}'")
        return

    # 2. Override config with command-line arguments
    config["noise"] = args.noise_level
    config["mes_level"]["turb_wd"] = not args.disable_turbine_wd
    config["mes_level"]["farm_wd"] = not args.disable_farm_wd

    print("--- Using Configuration ---")
    print(yaml.dump(config, indent=2))
    print("---------------------------")
    if args.noise_level == "Normal":
        print("INFO: WindGym's built-in 'Normal' noise will be applied.")
    if args.disable_turbine_wd:
        print("INFO: Turbine wind direction measurements DISABLED.")
    if args.disable_farm_wd:
        print("INFO: Farm wind direction measurements DISABLED.")

    # 3. W&B and Output Directory Setup
    run_name = f"{args.run_name_prefix}_noise-{args.noise_level}_turbWD-{not args.disable_turbine_wd}_farmWD-{not args.disable_farm_wd}_{wandb.util.generate_id()}"

    run = wandb.init(
        project=args.project_name,
        config={**vars(args), "final_config": config}, # Log all args and final config
        name=run_name,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    models_save_base = os.path.join(args.models_base_dir, run.id)
    tensorboard_log_base = os.path.join(args.tensorboard_base_dir, run.id)
    os.makedirs(models_save_base, exist_ok=True)
    os.makedirs(tensorboard_log_base, exist_ok=True)

    # Ensure TurbBox path is valid
    turbbox_path_train = args.turbbox_path
    if not os.path.isfile(turbbox_path_train) and args.turbtype == "MannLoad":
        potential_path_cwd = os.path.join(os.getcwd(), turbbox_path_train)
        if os.path.isfile(potential_path_cwd):
            turbbox_path_train = potential_path_cwd
            print(f"Using TurbBox found in CWD: {turbbox_path_train}")
        else:
            print(f"ERROR: Turbulence box file not found at '{turbbox_path_train}'.")
            exit()

    env_init_params = {
        "dt_sim": args.dt_sim,
        "dt_env": args.dt_env,
        "yaw_step_sim": args.yaw_step,
        "turbtype": args.turbtype,
        "TurbBox": turbbox_path_train,
        "n_passthrough": args.n_passthrough,
        "fill_window": args.fill_window,
    }

    turbine = V80()
    temp_yaml_filepath = None

    try:
        # 4. Create a temporary YAML file for the subprocesses to use
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmp_file:
            yaml.dump(config, tmp_file)
            temp_yaml_filepath = tmp_file.name
        print(f"Temporary YAML config for subprocesses: {temp_yaml_filepath}")

        # 5. Create the vectorized environment
        vec_env = SubprocVecEnv(
            [
                make_env(temp_yaml_filepath, turbine, env_init_params, seed=args.seed + i)
                for i in range(args.n_envs)
            ]
        )

        # 6. Setup Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=os.path.join(models_save_base, "checkpoints"),
            name_prefix="yaw_agent_model",
        )

        eval_env_instance = make_env(
            temp_yaml_filepath, turbine, env_init_params, seed=args.seed + 1000
        )()
        
        eval_callback = EvalCallback(
            eval_env_instance,
            best_model_save_path=os.path.join(models_save_base, "best_model"),
            log_path=os.path.join(models_save_base, "eval_logs"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            deterministic=True,
            render=False,
        )

        all_callbacks = [
            WandbCallback(
                gradient_save_freq=0,
                model_save_path=os.path.join(models_save_base, "wandb_models"),
                model_save_freq=max(args.save_freq // args.n_envs, 1),
                verbose=2,
            ),
            checkpoint_callback,
            eval_callback,
            WindGymCustomMonitor(),
        ]

        policy_kwargs = dict(
            net_arch=dict(
                pi=[int(x) for x in args.net_arch_pi.split(",")],
                vf=[int(x) for x in args.net_arch_vf.split(",")],
            )
        )
        
        # 7. Initialize and Train the PPO Model
        model = PPO(
            args.policy_type,
            vec_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log_base,
            seed=args.seed,
        )

        print(f"Starting training for run: {run.name}")
        model.learn(total_timesteps=args.total_timesteps, callback=all_callbacks)

        model.save(os.path.join(models_save_base, "final_model"))
        print(f"Final model saved to: {os.path.join(models_save_base, 'final_model.zip')}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 8. Cleanup
        if "vec_env" in locals():
            vec_env.close()
        if "eval_env_instance" in locals():
            eval_env_instance.close()
        if temp_yaml_filepath and os.path.exists(temp_yaml_filepath):
            os.remove(temp_yaml_filepath)
            print(f"Temporary YAML file {temp_yaml_filepath} deleted.")
        if "run" in locals() and run.id:
            run.finish()
        print("Training script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for WindGym.")
    
    # --- New/Modified Arguments ---
    parser.add_argument(
        "--config-path",
        type=str,
        default="config.yaml",
        help="Path to the base YAML configuration file."
    )

    # --- Experiment Setup ---
    parser.add_argument("--project_name", type=str, default="WindGym_Parameterized_Training", help="WandB project name.")
    parser.add_argument("--run_name_prefix", type=str, default="PPO_WindGym", help="Prefix for WandB run name.")
    parser.add_argument("--models_base_dir", type=str, default="./trained_models_param", help="Base directory to save models.")
    parser.add_argument("--tensorboard_base_dir", type=str, default="./sb3_tensorboard_logs_param", help="Base directory for SB3 TensorBoard logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # --- YAML/Observation Customization ---
    parser.add_argument("--noise_level", type=str, choices=["None", "Normal"], default="None", help="Override the 'noise' setting in the YAML config.")
    parser.add_argument("--disable_turbine_wd", action="store_true", help="Disable turbine wind direction measurements (overrides YAML).")
    parser.add_argument("--disable_farm_wd", action="store_true", help="Disable farm wind direction measurements (overrides YAML).")

    # --- WindFarmEnv Direct Parameters ---
    parser.add_argument("--dt_sim", type=float, default=1.0, help="DWM simulation timestep (s).")
    parser.add_argument("--dt_env", type=float, default=10.0, help="Agent environment timestep (s).")
    parser.add_argument("--yaw_step", type=float, default=1.0, help="Max yaw change per env step (degrees).")
    parser.add_argument("--turbtype", type=str, default="MannLoad", choices=["MannLoad", "MannGenerate", "MannFixed", "Random", "None"], help="Turbulence type.")
    parser.add_argument("--turbbox_path", type=str, default="Hipersim_mann_l5.0_ae1.0000_g0.0_h0_128x128x128_3.000x3.00x3.00_s0001.nc", help="Path to turbulence box file (if MannLoad).")
    parser.add_argument("--n_passthrough", type=int, default=10, help="Number of passthroughs for episode length calculation.")
    parser.add_argument("--fill_window", type=lambda x: (str(x).lower() == "true"), default=True, help="Fill observation window at reset (True/False).")

    # --- PPO Hyperparameters ---
    parser.add_argument("--policy_type", type=str, default="MlpPolicy", help="Policy type for PPO.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timesteps for training.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per environment per PPO update.")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for PPO.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs when optimizing the surrogate loss.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for GAE.")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient.")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping.")
    parser.add_argument("--net_arch_pi", type=str, default="128,128", help="Policy network architecture (comma-separated ints).")
    parser.add_argument("--net_arch_vf", type=str, default="128,128", help="Value network architecture (comma-separated ints).")

    # --- Callback Frequencies ---
    parser.add_argument("--save_freq", type=int, default=50_000, help="Frequency to save checkpoints (global steps).")
    parser.add_argument("--eval_freq", type=int, default=25_000, help="Frequency to run evaluations (global steps).")

    args = parser.parse_args()
    train_agent(args)
