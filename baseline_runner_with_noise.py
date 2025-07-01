import numpy as np
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import config as cfg
from Agent import Agent
from tqdm import tqdm
import argparse
from base_simulation import Simulation
import random

class NoisyAgent(Agent):
    """
    Agent that can misidentify neighbor types with a given probability (noise).
    """
    
    def __init__(self, type_id, noise_probability=0.0):
        super().__init__(type_id)
        self.noise_probability = noise_probability
    
    def _unlike_ratio_with_noise(self, r, c, grid):
        """
        Calculate unlike ratio with noise - agents may misidentify neighbor types.
        
        Args:
            r (int): Current row position
            c (int): Current column position
            grid (list): 2D grid representing the environment
            
        Returns:
            float: Ratio of unlike neighbors (potentially with misidentification)
        """
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r_n, c_n = r + dr, c + dc
                if 0 <= r_n < cfg.GRID_SIZE and 0 <= c_n < cfg.GRID_SIZE:
                    agent = grid[r_n][c_n]
                    if agent is not None:
                        neighbors.append(agent)
        
        if not neighbors:
            return 0  # No neighbors means no need to move
        
        # Count unlike neighbors with potential misidentification
        unlike_count = 0
        for neighbor in neighbors:
            # Determine perceived type (with noise)
            perceived_type = neighbor.type_id
            
            # Apply noise: with probability noise_probability, flip the perceived type
            if random.random() < self.noise_probability:
                perceived_type = 1 - neighbor.type_id  # Flip between 0 and 1
            
            # Count as unlike if perceived type differs from agent's type
            if perceived_type != self.type_id:
                unlike_count += 1
        
        return unlike_count / len(neighbors)
    
    def _unlike_ratio(self, r, c, grid):
        """Override the base _unlike_ratio method to use noisy perception."""
        return self._unlike_ratio_with_noise(r, c, grid)


def mechanical_decision_with_noise(agent, r, c, grid):
    """Decision function for noisy mechanical agents."""
    return agent.random_response(r, c, grid)


class NoisyBaselineSimulation(Simulation):
    """
    Baseline simulation with noisy agents that can misidentify neighbor types.
    """
    
    def __init__(self, run_id, noise_probability=0.0, config_override=None):
        self.noise_probability = noise_probability
        if config_override:
            for key, value in config_override.items():
                setattr(cfg, key, value)
        
        # Create agent factory that produces noisy agents
        def noisy_agent_factory(type_id):
            return NoisyAgent(type_id, noise_probability=self.noise_probability)
        
        super().__init__(run_id, agent_factory=noisy_agent_factory, 
                         decision_func=mechanical_decision_with_noise)


def run_single_simulation_with_noise(args):
    """Run a single simulation with noise."""
    run_id, noise_probability, config_override, output_dir = args
    sim = NoisyBaselineSimulation(run_id, noise_probability=noise_probability, 
                                 config_override=config_override)
    return sim.run_single_simulation(output_dir=output_dir, max_steps=1000)


def run_noisy_baseline_experiment(n_runs=100, max_steps=1000, noise_probability=0.0, 
                                 config_override=None, parallel=True):
    """
    Run baseline experiment with noise.
    
    Args:
        n_runs (int): Number of simulation runs
        max_steps (int): Maximum steps per simulation
        noise_probability (float): Probability that an agent misidentifies a neighbor's type (0.0 to 1.0)
        config_override (dict): Configuration overrides
        parallel (bool): Whether to run simulations in parallel
        
    Returns:
        tuple: (output_directory, results)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"noisy_baseline_{timestamp}"
    output_dir = f"experiments/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration dictionary
    config_dict = {
        'n_runs': n_runs,
        'max_steps': max_steps,
        'noise_probability': noise_probability,
        'grid_size': cfg.GRID_SIZE,
        'num_type_a': cfg.NUM_TYPE_A,
        'num_type_b': cfg.NUM_TYPE_B,
        'similarity_threshold': cfg.SIMILARITY_THRESHOLD,
        'agent_satisfaction_threshold': cfg.AGENT_SATISFACTION_THRESHOLD,
        'no_move_threshold': cfg.NO_MOVE_THRESHOLD,
        'timestamp': timestamp,
        'experiment_type': 'noisy_baseline'
    }
    
    if config_override:
        config_dict['overrides'] = config_override
    
    # Save configuration
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Prepare arguments for parallel processing
    args_list = [(i, noise_probability, config_override, output_dir) for i in range(n_runs)]
    
    # Run simulations
    if parallel:
        n_processes = min(cpu_count(), n_runs)
        with Pool(n_processes) as pool:
            results = list(tqdm(
                pool.imap(run_single_simulation_with_noise, args_list),
                total=n_runs,
                desc=f"Running noisy baseline simulations (noise={noise_probability:.3f})"
            ))
    else:
        results = []
        for args in tqdm(args_list, desc=f"Running noisy baseline simulations (noise={noise_probability:.3f})"):
            results.append(run_single_simulation_with_noise(args))

    # Analyze results
    output_dir, results, convergence_data = Simulation.analyze_results(results, output_dir, n_runs)
    
    # Print summary
    print(f"\nNoisy Baseline Experiment completed. Results saved to: {output_dir}")
    print(f"Noise probability: {noise_probability:.3f}")
    print(f"Total runs: {n_runs}")
    print(f"Converged runs: {sum(1 for r in convergence_data if r['converged'])}")
    
    converged_steps = [r['convergence_step'] for r in convergence_data if r['convergence_step'] is not None]
    if converged_steps:
        print(f"Average convergence step: {np.mean(converged_steps):.2f}")
    else:
        print("No runs converged")
    
    return output_dir, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run noisy baseline Schelling segregation simulations")
    parser.add_argument('--runs', type=int, default=100, 
                       help='Number of simulation runs (default: 100)')
    parser.add_argument('--max-steps', type=int, default=1000, 
                       help='Maximum steps per simulation (default: 1000)')
    parser.add_argument('--noise', type=float, default=0.1, 
                       help='Noise probability - chance agent misidentifies neighbor type (default: 0.1)')
    parser.add_argument('--no-parallel', action='store_true', 
                       help='Disable parallel processing')
    
    args = parser.parse_args()
    
    # Validate noise probability
    if not 0.0 <= args.noise <= 1.0:
        raise ValueError("Noise probability must be between 0.0 and 1.0")
    
    print(f"Starting noisy baseline experiment with {args.runs} runs and noise probability {args.noise}")
    
    run_noisy_baseline_experiment(
        n_runs=args.runs,
        max_steps=args.max_steps,
        noise_probability=args.noise,
        parallel=not args.no_parallel
    )