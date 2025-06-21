import argparse
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv, PyWakeAgent
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.utils.generate_layouts import generate_square_grid


class SB3AgentEvalWrapper:
    """Wraps a trained SB3 PPO agent."""
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)
        self.name = Path(model_path).stem

    def predict(self, observation, deterministic=True):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action, None

    def update_wind(self, **kwargs):
        pass


def load_agents_for_config(agent_dir: Path, config_name: str, x_pos, y_pos, include_pywake=True):
    agents = {}
    config_agents_path = agent_dir / config_name
    if not config_agents_path.exists():
        print(f"[!] No agents found for config '{config_name}' in {config_agents_path}")
        return agents

    for zip_file in config_agents_path.glob("*.zip"):
        agents[zip_file.stem] = SB3AgentEvalWrapper(str(zip_file))

    if include_pywake:
        agents["pywake"] = PyWakeAgent(x_pos=x_pos, y_pos=y_pos)

    return agents


def evaluate_config(config_name: str, configs_dir: Path, agents_dir: Path, out_dir: Path,
                    n_episodes: int, n_passthrough: float, include_pywake=True):
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return

    print(f"\n=== Evaluating config: {config_name} ===")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    turbine = V80()
    nx = config["farm"]["nx"]
    ny = config["farm"].get("ny", 1)
    xDist = config["farm"]["xDist"]
    yDist = config["farm"]["yDist"]

    x_pos, y_pos = generate_square_grid(turbine=turbine, nx=nx, ny=ny, xDist=xDist, yDist=yDist)

    def env_factory():
        return WindFarmEnv(
            turbine=turbine,
            x_pos=x_pos,
            y_pos=y_pos,
            yaml_path=str(config_path),
            turbtype="None",
            Baseline_comp=True,
        )

    agents = load_agents_for_config(agents_dir, config_name, x_pos, y_pos, include_pywake=include_pywake)
    if not agents:
        print(f"[!] Skipping {config_name}: no agents found.")
        return

    coliseum = Coliseum(env_factory=env_factory, agents=agents, n_passthrough=n_passthrough)
    _ = coliseum.run_time_series_evaluation(num_episodes=n_episodes, save_detailed_history=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary plot
    coliseum.plot_summary_comparison(save_path=str(out_dir / f"plot_{config_name}.png"))

    # Process and plot detailed time series results
    if hasattr(coliseum, 'time_series_results') and coliseum.time_series_results:
        all_dfs = []
        
        # Create figure for combined plots
        n_agents = len(coliseum.time_series_results)
        fig, axes = plt.subplots(4, n_agents, figsize=(6*n_agents, 16))
        if n_agents == 1:
            axes = axes.reshape(-1, 1)
        
        for agent_idx, (agent_name, episode_results) in enumerate(coliseum.time_series_results.items()):
            agent_data = {
                'time': [],
                'power': [],
                'power_baseline': [],
                'yaw_angles': [],
                'wind_speed': [],
                'wind_direction': [],
                'rewards': [],
                'cumulative_rewards': []
            }
            
            # Process all episodes for this agent
            for episode_data in episode_results:
                history = episode_data['history']
                
                # Extract time steps
                time_steps = history['step'] * config.get('dt_env', 1)  # Convert to seconds
                agent_data['time'].extend(time_steps)
                
                # Extract rewards
                agent_data['rewards'].extend(history['reward'])
                agent_data['cumulative_rewards'].extend(history['mean_cumulative_reward'])
                
                # Extract data from info dictionaries
                for info in history['info']:
                    # Power data
                    if 'Power agent' in info:
                        agent_data['power'].append(info['Power agent'])
                    if 'Power baseline' in info:
                        agent_data['power_baseline'].append(info['Power baseline'])
                    
                    # Yaw angles (average across turbines)
                    if 'yaw angles agent' in info:
                        agent_data['yaw_angles'].append(np.mean(info['yaw angles agent']))
                    
                    # Wind conditions
                    if 'Wind speed Global' in info:
                        agent_data['wind_speed'].append(info['Wind speed Global'])
                    if 'Wind direction Global' in info:
                        agent_data['wind_direction'].append(info['Wind direction Global'])
            
            # Convert to numpy arrays for easier manipulation
            for key in agent_data:
                agent_data[key] = np.array(agent_data[key])
            
            # Create DataFrame for this agent
            agent_df = pd.DataFrame({
                't': agent_data['time'],
                'power': agent_data['power'],
                'power_baseline': agent_data['power_baseline'] if len(agent_data['power_baseline']) > 0 else np.nan,
                'yaw_mean': agent_data['yaw_angles'],
                'wind_speed': agent_data['wind_speed'],
                'wind_direction': agent_data['wind_direction'],
                'reward': agent_data['rewards'],
                'cumulative_reward': agent_data['cumulative_rewards'],
                'agent': agent_name
            })
            all_dfs.append(agent_df)
            
            # Plot 1: Power comparison
            ax = axes[0, agent_idx]
            ax.plot(agent_data['time'], agent_data['power']/1e6, label='Agent', linewidth=2)
            if len(agent_data['power_baseline']) > 0:
                ax.plot(agent_data['time'], agent_data['power_baseline']/1e6, 
                       label='Baseline', linewidth=2, linestyle='--')
            ax.set_ylabel('Power [MW]')
            ax.set_title(f'{agent_name} - Power Output')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Power gain percentage
            ax = axes[1, agent_idx]
            if len(agent_data['power_baseline']) > 0:
                power_gain = (agent_data['power'] - agent_data['power_baseline']) / agent_data['power_baseline'] * 100
                ax.plot(agent_data['time'], power_gain, color='green', linewidth=2)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax.set_ylabel('Power Gain [%]')
                ax.set_title('Power Gain vs Baseline')
            else:
                ax.text(0.5, 0.5, 'No baseline data', ha='center', va='center', transform=ax.transAxes)
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Yaw angles and wind direction
            ax = axes[2, agent_idx]
            ax.plot(agent_data['time'], agent_data['yaw_angles'], label='Mean Yaw', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(agent_data['time'], agent_data['wind_direction'], 
                    color='red', alpha=0.5, label='Wind Dir', linewidth=1)
            ax.set_ylabel('Yaw Angle [deg]')
            ax2.set_ylabel('Wind Direction [deg]', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax.set_title('Yaw Control')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Cumulative reward
            ax = axes[3, agent_idx]
            ax.plot(agent_data['time'], agent_data['cumulative_rewards'], 
                   color='purple', linewidth=2)
            ax.set_ylabel('Mean Cumulative Reward')
            ax.set_xlabel('Time [s]')
            ax.set_title('Learning Performance')
            ax.grid(True, alpha=0.3)
            
            # Save individual agent detailed plot
            fig_agent = plt.figure(figsize=(12, 8))
            gs = fig_agent.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Subplot 1: Power time series
            ax1 = fig_agent.add_subplot(gs[0, :])
            ax1.plot(agent_data['time'], agent_data['power']/1e6, label='Agent Power', linewidth=2)
            if len(agent_data['power_baseline']) > 0:
                ax1.plot(agent_data['time'], agent_data['power_baseline']/1e6, 
                        label='Baseline Power', linewidth=2, linestyle='--')
                ax1.fill_between(agent_data['time'], 
                               agent_data['power']/1e6, 
                               agent_data['power_baseline']/1e6,
                               where=(agent_data['power'] > agent_data['power_baseline']),
                               alpha=0.3, color='green', label='Gain')
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Power [MW]')
            ax1.set_title(f'{agent_name} - Power Production Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Power gain histogram
            ax2 = fig_agent.add_subplot(gs[1, 0])
            if len(agent_data['power_baseline']) > 0:
                power_gain = (agent_data['power'] - agent_data['power_baseline']) / agent_data['power_baseline'] * 100
                ax2.hist(power_gain, bins=50, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(x=0, color='red', linestyle='--', label='No gain')
                ax2.axvline(x=np.mean(power_gain), color='blue', linestyle='-', 
                           label=f'Mean: {np.mean(power_gain):.2f}%')
                ax2.set_xlabel('Power Gain [%]')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Power Gain Distribution')
                ax2.legend()
            
            # Subplot 3: Reward analysis
            ax3 = fig_agent.add_subplot(gs[1, 1])
            ax3.plot(agent_data['time'], agent_data['rewards'], alpha=0.5, label='Instant Reward')
            ax3.plot(agent_data['time'], agent_data['cumulative_rewards'], 
                    linewidth=2, label='Mean Cumulative')
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Reward')
            ax3.set_title('Reward Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Subplot 4: Wind conditions
            ax4 = fig_agent.add_subplot(gs[2, :])
            ax4.plot(agent_data['time'], agent_data['wind_speed'], label='Wind Speed')
            ax4.set_xlabel('Time [s]')
            ax4.set_ylabel('Wind Speed [m/s]', color='blue')
            ax4.tick_params(axis='y', labelcolor='blue')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(agent_data['time'], agent_data['wind_direction'], 
                         color='red', alpha=0.7, label='Wind Direction')
            ax4_twin.set_ylabel('Wind Direction [deg]', color='red')
            ax4_twin.tick_params(axis='y', labelcolor='red')
            ax4.set_title('Wind Conditions During Evaluation')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_agent.savefig(out_dir / f"detailed_timeseries_{config_name}_{agent_name}.png", dpi=150)
            plt.close(fig_agent)
        
        # Save combined plot
        fig.suptitle(f'Agent Comparison - {config_name}', fontsize=16)
        plt.tight_layout()
        fig.savefig(out_dir / f"agent_comparison_{config_name}.png", dpi=150)
        plt.close(fig)
        
        # Save combined DataFrame
        if all_dfs:
            full_df = pd.concat(all_dfs, ignore_index=True)
            full_df.to_csv(out_dir / f"timeseries_{config_name}.csv", index=False)
            
            # Generate summary statistics
            summary_stats = full_df.groupby('agent').agg({
                'power': ['mean', 'std', 'min', 'max'],
                'cumulative_reward': ['mean', 'std', 'min', 'max']
            }).round(2)
            summary_stats.to_csv(out_dir / f"summary_stats_{config_name}.csv")
            
            print(f"\nSummary Statistics:")
            print(summary_stats)
    
    print(f"[✓] Evaluation complete for {config_name}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate WindGym agents for a specific config.")
    parser.add_argument("--config-name", type=str, required=True, help="Config name (without .yaml)")
    parser.add_argument("--configs-dir", type=str, default="configs/", help="Path to config YAMLs")
    parser.add_argument("--agents-dir", type=str, default="agents/", help="Path to agent zip files")
    parser.add_argument("--out-dir", type=str, default="results/", help="Where to store output")
    parser.add_argument("--n-episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--n-passthrough", type=float, default=6.0, help="Passthrough value")
    parser.add_argument("--no-pywake", action="store_true", help="Exclude PyWake baseline")
    args = parser.parse_args()

    evaluate_config(
        config_name=args.config_name,
        configs_dir=Path(args.configs_dir),
        agents_dir=Path(args.agents_dir),
        out_dir=Path(args.out_dir),
        n_episodes=args.n_episodes,
        n_passthrough=args.n_passthrough,
        include_pywake=not args.no_pywake,
    )


if __name__ == "__main__":
    main()

