"""
Parallel Tempering MCMC Optimization of J/P_0
Multiple chains at FIXED temperatures run in parallel
Starting from V = 0 with gradient constraint: -1 < ∂V/∂x < 1
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
import numpy as np

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

class ParallelTemperingOptimizer:
    """
    Parallel Tempering MCMC for optimizing V(x) to maximize J/P_0
    
    Runs 5 chains at different FIXED temperatures simultaneously
    Enforces gradient constraint: -1 < ∂V/∂x < 1
    """
    
    def __init__(self, L=1.0, alpha=1.0, n_points=30, temperatures=[0.01, 0.1, 0.5, 1.0, 2.0]):
        """
        Parameters:
        -----------
        L : float
            System length
        alpha : float
            Model parameter
        n_points : int
            Number of spatial discretization points
        temperatures : list
            Fixed temperatures for parallel chains
        """
        self.L = L
        self.alpha = alpha
        self.n_points = n_points
        self.x = jnp.linspace(0, L, n_points)
        self.dx = L / (n_points - 1)
        self.temperatures = temperatures
        self.n_chains = len(temperatures)
        
        # MCMC tracking for each temperature
        self.V_history = {T: [] for T in temperatures}
        self.J_history = {T: [] for T in temperatures}
        self.acceptance_history = {T: [] for T in temperatures}
        
    def compute_derivative(self, V):
        """Compute spatial derivative using finite differences"""
        return jnp.gradient(V, self.dx)
    
    def check_gradient_constraint(self, V):
        """
        Check if gradient constraint is satisfied: -1 < ∂V/∂x < 1
        
        This avoids singularity in J/P_0 formula where V² - (∂V)² appears in denominator
        """
        dV = self.compute_derivative(V)
        return jnp.all(dV > -1.0) and jnp.all(dV < 1.0)
    
    def compute_xi(self, V, a_idx, b_idx):
        """
        Compute ξ_{a,b} = exp[∫_a^b dc α∂_c V / (V^2 - (∂_c V)^2)]
        """
        if a_idx >= b_idx:
            return 1.0
            
        dV = self.compute_derivative(V)
        denominator = V**2 - dV**2
        
        # With gradient constraint |dV| < 1, we need V > 1 to avoid issues
        # Add regularization
        denominator = jnp.where(jnp.abs(denominator) < 1e-8, 
                                jnp.sign(denominator) * 1e-8, 
                                denominator)
        
        integrand = self.alpha * dV / denominator
        integral = jnp.trapezoid(integrand[a_idx:b_idx+1], self.x[a_idx:b_idx+1])
        return jnp.exp(jnp.clip(integral, -50, 50))
    
    def compute_simplified_current(self, V):
        """
        Compute J/P_0 using simplified formula:
        J/p(0) = [1 - ξ_{0,L}] / [∫ (∂V - α) · ξ_{0,x} dx]
        
        With gradient constraint: -1 < ∂V/∂x < 1
        """
        # Check gradient constraint FIRST
        if not self.check_gradient_constraint(V):
            return -1e10  # Reject if constraint violated
        
        dV = self.compute_derivative(V)
        
        # Check for pathological V
        if jnp.all(jnp.abs(V) < 0.01):
            return -1e2  # Reject V≈0 everywhere
        
        if jnp.any(V < 0.1) or jnp.any(V > 10.0):
            return -1e2  # Keep V bounded
        
        # Numerator: 1 - ξ_{0,L}
        xi_0L = self.compute_xi(V, 0, self.n_points-1)
        
        if not jnp.isfinite(xi_0L):
            return -1e10
        
        numerator = 1.0 - xi_0L
        
        # Denominator: ∫ [(∂V - α)] · ξ_{0,x} dx
        denominator_integrand = jnp.zeros(self.n_points)
        for i in range(self.n_points):
            xi_0x = self.compute_xi(V, 0, i)
            
            if not jnp.isfinite(xi_0x):
                return -1e10
            
            V_safe = jnp.where(jnp.abs(V[i]) < 1e-8, 1e-8, V[i])
            denominator_integrand = denominator_integrand.at[i].set(
                (dV[i] - self.alpha) * xi_0x
            )
        
        denominator = jnp.trapezoid(denominator_integrand, self.x)
        denominator = jnp.where(jnp.abs(denominator) < 1e-10,
                               jnp.sign(denominator) * 1e-10,
                               denominator)
        
        result = numerator / denominator
        
        # Safeguard against explosion
        if not jnp.isfinite(result) or jnp.abs(result) > 1e6:
            return -1e10
        
        return result
    
    def propose_perturbation(self, V_current, perturbation_scale=0.05, n_modes=3):
        max_attempts = 10
        
        for attempt in range(max_attempts):
            # Check if starting from zero
            if np.all(np.abs(V_current) < 0.01):
                scale = perturbation_scale * 5.0  # Larger initial perturbation
            else:
                scale = perturbation_scale
            
            # Generate smooth perturbation using Fourier modes
            perturbation = np.zeros(self.n_points)
            
            for k in range(1, n_modes + 1):
                amplitude = scale * np.exp(-0.8 * k)
                phase = np.random.uniform(0, 2*np.pi)
                perturbation += amplitude * np.sin(2*np.pi*k*self.x/self.L + phase)
            
            # Add small noise
            perturbation += scale * 0.1 * np.random.randn(self.n_points)
            
            V_proposed = V_current + perturbation
            
            # Keep V positive and bounded
            V_proposed = np.clip(V_proposed, 0.1, 5.0)
            
            # Check gradient constraint
            V_jnp = jnp.array(V_proposed)
            if self.check_gradient_constraint(V_jnp):
                return V_jnp
        
        # If all attempts fail, return current V (rejection)
        return V_current
    
    def metropolis_step(self, V_current, J_current, temperature, perturbation_scale):
        """
        A single Metropolis-Hastings step at fixed temperature
        """
        # Propose new V
        V_proposed = self.propose_perturbation(V_current, perturbation_scale)
        
        # Check if proposal was rejected due to constraint
        if jnp.allclose(V_proposed, V_current):
            return V_current, J_current, False
        
        # Evaluate objective
        try:
            J_proposed = float(self.compute_simplified_current(V_proposed))
            
            # Metropolis criterion at FIXED temperature
            delta_J = J_proposed - J_current
            
            if delta_J > 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(delta_J / temperature)
            
            if np.random.rand() < accept_prob:
                return V_proposed, J_proposed, True
            else:
                return V_current, J_current, False
                
        except:
            return V_current, J_current, False
    
    def run_parallel_tempering(self, n_iterations=5000, V_init=None, 
                               perturbation_scale=0.1, save_every=10):
        """
        Run parallel markov chains with fixed temperatures Each chain runs independently at its own fixed temperature.
        """
        # Initialize all chains from same V_init
        if V_init is None:
            V_init = np.zeros(self.n_points)
        
        # State for each chain
        V_chains = {T: jnp.array(V_init) for T in self.temperatures}
        J_chains = {T: -1e10 for T in self.temperatures}
        
        # Initialize J for each chain
        for T in self.temperatures:
            try:
                J_chains[T] = float(self.compute_simplified_current(V_chains[T]))
            except:
                J_chains[T] = -1e10
        
        # Best overall
        V_best = V_chains[self.temperatures[0]].copy()
        J_best = J_chains[self.temperatures[0]]
        
        # Reset history
        for T in self.temperatures:
            self.V_history[T] = [np.array(V_chains[T])]
            self.J_history[T] = [J_chains[T]]
            self.acceptance_history[T] = []
        
        n_accepted = {T: 0 for T in self.temperatures}
        
        print(f"\n" + "="*80)
        print(f"PARALLEL HMMC")
        print(f"Running {self.n_chains} chains at FIXED temperatures")
        print(f"Temperatures: {self.temperatures}")
        print(f"Gradient constraint: -1 < ∂V/∂x < 1")
        print(f"Initial V: zeros (flat potential)")
        print("="*80)
        
        # Main MCMC loop
        for i in range(n_iterations):
            # Update each chain independently
            for T in self.temperatures:
                V_chains[T], J_chains[T], accepted = self.metropolis_step(
                    V_chains[T], J_chains[T], T, perturbation_scale
                )
                
                if accepted:
                    n_accepted[T] += 1
                    
                    # Update global best
                    if J_chains[T] > J_best:
                        V_best = V_chains[T].copy()
                        J_best = J_chains[T]
            
            # Save history
            if i % save_every == 0:
                for T in self.temperatures:
                    self.V_history[T].append(np.array(V_chains[T]))
                    self.J_history[T].append(J_chains[T])
                    acceptance_rate = n_accepted[T] / (i + 1)
                    self.acceptance_history[T].append(acceptance_rate)
            
            # Progress report
            if (i+1) % 1000 == 0:
                print(f"\nIteration {i+1}/{n_iterations}:")
                for T in self.temperatures:
                    acc_rate = n_accepted[T] / (i + 1)
                    print(f"  T={T:5.2f}: J={J_chains[T]:12.4f}, acceptance={acc_rate:.3f}")
                print(f"  Best overall: J={J_best:.4f}")
                print("-"*80)
        
        print("\n" + "="*80)
        print(f"Best J/P₀ found: {J_best:.6f}")
        print(f"\nFinal acceptance rates:")
        for T in self.temperatures:
            final_acc = n_accepted[T] / n_iterations
            print(f"  T={T:5.2f}: {final_acc:.3f}")
        print("="*80)
        
        return V_best, J_best
    
    def plot_all_temperatures(self, V_init, V_best):
        """
        Plot results for all temperatures on same plots
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # Define colors for each temperature
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        temp_colors = {T: colors[i] for i, T in enumerate(self.temperatures)}
        
        # Plot 1: J/P_0 traces for all temperatures
        ax1 = fig.add_subplot(gs[0, :])
        for T in self.temperatures:
            iterations = np.arange(len(self.J_history[T])) * 10
            ax1.plot(iterations, self.J_history[T], '-', 
                    color=temp_colors[T], linewidth=2, alpha=0.7, label=f'T={T}')
        
        ax1.set_xlabel('Iteration', fontsize=13)
        ax1.set_ylabel('J/P₀ (Current)', fontsize=13)
        ax1.set_title('Current Evolution at All Temperatures (Fixed T per chain)', 
                     fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, ncol=5, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Running maximum for all temperatures
        ax2 = fig.add_subplot(gs[1, :])
        for T in self.temperatures:
            iterations = np.arange(len(self.J_history[T])) * 10
            running_max = np.maximum.accumulate(self.J_history[T])
            ax2.plot(iterations, running_max, '-', 
                    color=temp_colors[T], linewidth=2.5, alpha=0.8, label=f'T={T}')
        
        ax2.set_xlabel('Iteration', fontsize=13)
        ax2.set_ylabel('Best J/P₀ So Far', fontsize=13)
        ax2.set_title('Running Maximum at All Temperatures', 
                     fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, ncol=5, loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Acceptance rates
        ax3 = fig.add_subplot(gs[2, 0])
        for T in self.temperatures:
            iterations = np.arange(len(self.acceptance_history[T])) * 10
            ax3.plot(iterations, self.acceptance_history[T], '-',
                    color=temp_colors[T], linewidth=2, alpha=0.7, label=f'T={T}')
        
        ax3.axhline(y=0.234, color='k', linestyle='--', alpha=0.3, label='Optimal')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Acceptance Rate', fontsize=12)
        ax3.set_title('Acceptance Rates (All T)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Final potentials from each temperature
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=2, label='V=0 initial')
        for T in self.temperatures:
            V_final = self.V_history[T][-1]
            ax4.plot(self.x, V_final, '-', color=temp_colors[T], 
                    linewidth=2.5, alpha=0.7, label=f'T={T}')
        
        ax4.set_xlabel('Position x', fontsize=12)
        ax4.set_ylabel('V(x)', fontsize=12)
        ax4.set_title('Final Potentials (All T)', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9, ncol=2)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Final gradients (check constraint)
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axhline(y=-1, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Constraint')
        ax5.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=2)
        ax5.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        
        for T in self.temperatures:
            V_final = self.V_history[T][-1]
            dV_final = self.compute_derivative(jnp.array(V_final))
            ax5.plot(self.x, dV_final, '-', color=temp_colors[T],
                    linewidth=2, alpha=0.7, label=f'T={T}')
        
        ax5.set_xlabel('Position x', fontsize=12)
        ax5.set_ylabel('∂V/∂x', fontsize=12)
        ax5.set_title('Final Gradients: -1 < ∂V/∂x < 1', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=9, ncol=2)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-1.2, 1.2)
        
        # Plot 6: Histogram of final J values
        ax6 = fig.add_subplot(gs[3, 0])
        final_J_values = [self.J_history[T][-1] for T in self.temperatures]
        bars = ax6.bar(range(len(self.temperatures)), final_J_values,
                      color=[temp_colors[T] for T in self.temperatures],
                      alpha=0.7, edgecolor='black', linewidth=2)
        ax6.set_xticks(range(len(self.temperatures)))
        ax6.set_xticklabels([f'T={T}' for T in self.temperatures], fontsize=10)
        ax6.set_ylabel('Final J/P₀', fontsize=12)
        ax6.set_title('Final Current by Temperature', fontsize=13, fontweight='bold')
        ax6.grid(True, axis='y', alpha=0.3)
        
        # Add values on bars
        for i, (T, val) in enumerate(zip(self.temperatures, final_J_values)):
            ax6.text(i, val, f'{val:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        
        # Plot 7: Evolution snapshots for one temperature (lowest T, best convergence)
        ax7 = fig.add_subplot(gs[3, 1:])
        T_best = min(self.temperatures)
        n_snapshots = min(len(self.V_history[T_best]), 10)
        indices = np.linspace(0, len(self.V_history[T_best])-1, n_snapshots, dtype=int)
        cmap = plt.cm.viridis
        
        for idx, i in enumerate(indices):
            color = cmap(idx / (len(indices) - 1))
            iter_num = i * 10
            alpha_val = 0.3 + 0.7 * (idx / (len(indices) - 1))
            linewidth = 1 + 2 * (idx / (len(indices) - 1))
            label = f'Iter {iter_num}' if idx % 2 == 0 else None
            ax7.plot(self.x, self.V_history[T_best][i], color=color,
                    alpha=alpha_val, linewidth=linewidth, label=label)
        
        ax7.set_xlabel('Position x', fontsize=12)
        ax7.set_ylabel('V(x)', fontsize=12)
        ax7.set_title(f'Evolution at T={T_best} (Coldest Chain)', 
                     fontsize=13, fontweight='bold')
        ax7.legend(fontsize=9, ncol=2)
        ax7.grid(True, alpha=0.3)
        
        fig.suptitle('Parallel HMMC: All Temperatures Comparison\n' + 
                    'Gradient Constraint: -1 < ∂V/∂x < 1',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current/HMMC_results.png', 
                   dpi=300, bbox_inches='tight')
        print("\nPlot saved: parallel_tempering_results.png")
        return fig
    
    def plot_diagnostics_all_temps(self):
        """
        Diagnostic plots showing all temperatures
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        temp_colors = {T: colors[i] for i, T in enumerate(self.temperatures)}
        
        # Plot 1: J distributions
        ax1 = axes[0, 0]
        for T in self.temperatures:
            ax1.hist(self.J_history[T], bins=30, alpha=0.4, 
                    color=temp_colors[T], label=f'T={T}', edgecolor='black')
        
        ax1.set_xlabel('J/P₀', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of J/P₀ at Each Temperature', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Exploration range (max - min) over time
        ax2 = axes[0, 1]
        for T in self.temperatures:
            iterations = np.arange(len(self.J_history[T])) * 10
            # Rolling max - min in windows
            window = 50
            if len(self.J_history[T]) > window:
                J_array = np.array(self.J_history[T])
                exploration = []
                for i in range(len(J_array) - window):
                    exploration.append(np.max(J_array[i:i+window]) - 
                                     np.min(J_array[i:i+window]))
                ax2.plot(iterations[:len(exploration)], exploration, '-',
                        color=temp_colors[T], linewidth=2, alpha=0.7, label=f'T={T}')
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Exploration Range (50-iter window)', fontsize=12)
        ax2.set_title('Exploration Magnitude by Temperature', 
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temperature vs final performance
        ax3 = axes[1, 0]
        final_means = [np.mean(self.J_history[T][-100:]) for T in self.temperatures]
        final_maxs = [np.max(self.J_history[T]) for T in self.temperatures]
        
        ax3.plot(self.temperatures, final_means, 'o-', markersize=10, 
                linewidth=3, color='blue', label='Mean (last 100)')
        ax3.plot(self.temperatures, final_maxs, 's-', markersize=10,
                linewidth=3, color='red', label='Maximum')
        
        ax3.set_xlabel('Temperature', fontsize=12)
        ax3.set_ylabel('J/P₀', fontsize=12)
        ax3.set_title('Final Performance vs Temperature', 
                     fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = "SUMMARY STATISTICS BY TEMPERATURE\n" + "="*50 + "\n\n"
        for T in self.temperatures:
            final_J = self.J_history[T][-1]
            max_J = np.max(self.J_history[T])
            mean_J = np.mean(self.J_history[T][-100:])
            final_acc = self.acceptance_history[T][-1]
            
            stats_text += f"T = {T:5.2f}:\n"
            stats_text += f"  Final J/P₀:  {final_J:10.4f}\n"
            stats_text += f"  Max J/P₀:    {max_J:10.4f}\n"
            stats_text += f"  Mean (last): {mean_J:10.4f}\n"
            stats_text += f"  Acceptance:  {final_acc:10.3f}\n\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current/HMMC_diagnostics.png',
                   dpi=300, bbox_inches='tight')
        print("Diagnostics saved: parallel_tempering_diagnostics.png")
        return fig


def main():
    """Run parallel tempering MCMC with fixed temperatures"""
    
    print("="*80)
    print("PARALLEL TEMPERING MCMC OPTIMIZATION")
    print("Multiple chains at FIXED temperatures")
    print("="*80)
    
    # Initialize with 5 temperatures
    temperatures = [0.01, 0.1, 0.5, 1.0, 2.0]
    
    # Setting constant
    optimizer = ParallelTemperingOptimizer(
        L=1.0,
        alpha=1.0,
        n_points=30,
        temperatures=temperatures
    )
    
    # Initial potential
    V_init = np.zeros(optimizer.n_points)
    
    # Run parallel tempering
    V_best, J_best = optimizer.run_parallel_tempering(
        n_iterations=5000,
        V_init=V_init,
        perturbation_scale=0.1,
        save_every=10
    )
    
    # Plot results
    optimizer.plot_all_temperatures(V_init, V_best)
    
    print("\nGenerating diagnostics...")
    optimizer.plot_diagnostics_all_temps()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nBest potential found has:")
    print(f"  J/P₀ = {J_best:.6f}")
    print(f"  Mean V = {np.mean(V_best):.4f}")
    print(f"  Std V = {np.std(V_best):.4f}")
    
    # Check gradient constraint
    dV_best = optimizer.compute_derivative(V_best)
    print(f"\nGradient constraint check:")
    print(f"  min(∂V/∂x) = {np.min(dV_best):.4f} (must be > -1)")
    print(f"  max(∂V/∂x) = {np.max(dV_best):.4f} (must be < +1)")
    print(f"  Constraint satisfied: {np.all(dV_best > -1) and np.all(dV_best < 1)}")
    print("="*80)
    
    plt.show()
    
    return optimizer, V_best, J_best


if __name__ == "__main__":
    optimizer, V_best, J_best = main()
