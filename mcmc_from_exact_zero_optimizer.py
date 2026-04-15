"""
Metropolis-Hastings MCMC Optimization of J/P_0
Starting from TRULY FLAT POTENTIAL (V = 0 exactly)
with small random perturbations
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

class MCMCVariationalOptimizer:
    """
    Metropolis-Hastings MCMC for optimizing V(x) to maximize J/P_0
    
    Starts from V = 0 (exactly zero, truly flat) and explores via random perturbations
    """
    
    def __init__(self, L=1.0, alpha=1.0, n_points=30):
        """
        Parameters:
        -----------
        L : float
            System length
        alpha : float
            Model parameter
        n_points : int
            Number of spatial discretization points
        """
        self.L = L
        self.alpha = alpha
        self.n_points = n_points
        self.x = jnp.linspace(0, L, n_points)
        self.dx = L / (n_points - 1)
        
        # MCMC tracking
        self.V_history = []
        self.J_history = []
        self.acceptance_history = []
        self.temperature_history = []
        
    def compute_derivative(self, V):
        """Compute spatial derivative using finite differences"""
        return jnp.gradient(V, self.dx)
    
    def compute_second_derivative(self, V):
        """Compute second spatial derivative"""
        dV = self.compute_derivative(V)
        return jnp.gradient(dV, self.dx)
    
    def compute_xi(self, V, a_idx, b_idx):
        """
        Compute ξ_{a,b} = exp[∫_a^b dc α∂_c V / (V^2 - (∂_c V)^2)]
        """
        if a_idx >= b_idx:
            return 1.0
            
        dV = self.compute_derivative(V)
        denominator = V**2 - dV**2
        denominator = jnp.where(jnp.abs(denominator) < 1e-8, 
                                jnp.sign(denominator) * 1e-8, 
                                denominator)
        
        integrand = self.alpha * dV / denominator
        integral = jnp.trapezoid(integrand[a_idx:b_idx+1], self.x[a_idx:b_idx+1])
        return jnp.exp(jnp.clip(integral, -50, 50))
    
    def compute_simplified_current(self, V):
        """
        Compute J/P_0 using simplified formula:
        J/p(0) = [1 - ξ_{0,L}] / [∫ (∂V - α)/V² · ξ_{0,x} dx]
        
        This is faster for MCMC exploration
        
        Special handling for V=0 case:
        - V must be > 0 to avoid division by zero
        - First perturbation will move away from exactly zero
        """
        dV = self.compute_derivative(V)
        
        # Check for pathological V
        # Allow V to start at 0 but penalize if it stays there
        if jnp.all(jnp.abs(V) < 0.01):
            # All V values too close to zero - this is the initial flat state
            # Return very negative value to encourage exploration
            return -1e10
        
        if jnp.any(V < 0.01) or jnp.any(V > 10.0):
            return -1e10  # Reject bad potentials
        
        if jnp.any(jnp.abs(dV) > 50):
            return -1e10  # Reject crazy gradients
        
        # Numerator: 1 - ξ_{0,L}
        xi_0L = self.compute_xi(V, 0, self.n_points-1)
        
        # Check for numerical issues
        if not jnp.isfinite(xi_0L):
            return -1e10
        
        numerator = 1.0 - xi_0L
        
        # Denominator: ∫ [(∂V - α)/V²] · ξ_{0,x} dx
        denominator_integrand = jnp.zeros(self.n_points)
        for i in range(self.n_points):
            xi_0x = self.compute_xi(V, 0, i)
            
            if not jnp.isfinite(xi_0x):
                return -1e10
            
            # Add small epsilon to avoid exact division by zero
            V_safe = jnp.where(jnp.abs(V[i]) < 1e-8, 1e-8, V[i])
            denominator_integrand = denominator_integrand.at[i].set(
                (dV[i] - self.alpha) / (V_safe**2) * xi_0x
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
    
    def propose_perturbation(self, V_current, perturbation_scale=0.05, 
                            n_modes=3, smoothness=0.8):
        """
        Propose new V by adding smooth random perturbation
        
        Special handling for V=0 initial state:
        - First perturbation needs to break away from zero
        - Subsequent perturbations explore the landscape
        
        Parameters:
        -----------
        V_current : array
            Current potential
        perturbation_scale : float
            Amplitude of perturbation
        n_modes : int
            Number of Fourier modes in perturbation
        smoothness : float
            Smoothness parameter (higher = smoother)
        
        Returns:
        --------
        V_proposed : array
            Proposed new potential
        """
        # Check if we're starting from zero
        if np.all(np.abs(V_current) < 0.01):
            # Starting from flat V=0, need larger initial perturbation
            # to break symmetry
            perturbation_scale *= 5.0
        
        # Generate smooth perturbation using Fourier modes
        perturbation = np.zeros(self.n_points)
        
        for k in range(1, n_modes + 1):
            amplitude = perturbation_scale * np.exp(-smoothness * k)
            phase = np.random.uniform(0, 2*np.pi)
            perturbation += amplitude * np.sin(2*np.pi*k*self.x/self.L + phase)
        
        # Add small random noise
        perturbation += perturbation_scale * 0.1 * np.random.randn(self.n_points)
        
        V_proposed = V_current + perturbation
        
        # Keep V positive and bounded
        # For V=0 start, this will push it into positive range
        V_proposed = np.clip(V_proposed, 0.05, 5.0)
        
        return jnp.array(V_proposed)
    
    def metropolis_hastings_step(self, V_current, J_current, temperature=1.0,
                                  perturbation_scale=0.05):
        """
        Single Metropolis-Hastings step
        
        Returns:
        --------
        V_new : array
            Accepted or rejected potential
        J_new : float
            Current value
        accepted : bool
            Whether proposal was accepted
        """
        # Propose new V
        V_proposed = self.propose_perturbation(V_current, perturbation_scale)
        
        # Evaluate objective
        try:
            J_proposed = float(self.compute_simplified_current(V_proposed))
            
            # Metropolis-Hastings acceptance criterion
            # We want to MAXIMIZE J, so accept if J_proposed > J_current
            delta_J = J_proposed - J_current
            
            # Acceptance probability (simulated annealing style)
            if delta_J > 0:
                # Always accept improvements
                accept_prob = 1.0
            else:
                # Accept downgrades with probability exp(ΔJ/T)
                accept_prob = np.exp(delta_J / temperature)
            
            # Accept or reject
            if np.random.rand() < accept_prob:
                return V_proposed, J_proposed, True
            else:
                return V_current, J_current, False
                
        except:
            # If evaluation fails, reject
            return V_current, J_current, False
    
    def run_mcmc(self, n_iterations=5000, V_init=None, 
                 initial_temp=2.0, final_temp=0.01,
                 perturbation_scale=0.1, save_every=10):
        """
        Run MCMC optimization
        
        Parameters:
        -----------
        n_iterations : int
            Number of MCMC steps
        V_init : array, optional
            Initial potential (if None, starts at EXACTLY ZERO)
        initial_temp : float
            Starting temperature for simulated annealing (higher for V=0 start)
        final_temp : float
            Final temperature
        perturbation_scale : float
            Size of random perturbations
        save_every : int
            Save state every N iterations
        
        Returns:
        --------
        V_best : array
            Best potential found
        J_best : float
            Best current value
        """
        # Initialize from EXACTLY ZERO potential
        if V_init is None:
            # Start from EXACTLY V = 0 everywhere
            V_init = np.zeros(self.n_points)
        
        V_current = jnp.array(V_init)
        
        # Evaluate initial current (will be very bad for V=0)
        try:
            J_current = float(self.compute_simplified_current(V_current))
        except:
            J_current = -1e10
        
        # Best so far
        V_best = V_current.copy()
        J_best = J_current
        
        # Reset history
        self.V_history = []
        self.J_history = []
        self.acceptance_history = []
        self.temperature_history = []
        
        # Save initial state
        self.V_history.append(np.array(V_current))
        self.J_history.append(J_current)
        
        print(f"\n" + "="*70)
        print(f"Starting MCMC with {n_iterations} iterations")
        print(f"Initial potential: V(x) = 0 EXACTLY (truly flat, no structure)")
        print(f"Initial J/P₀ = {J_current:.6f} (expected to be very negative)")
        print(f"")
        print(f"The algorithm will spontaneously break this symmetry!")
        print("="*70)
        
        n_accepted = 0
        
        # MCMC loop
        for i in range(n_iterations):
            # Simulated annealing: decrease temperature
            temperature = initial_temp * (final_temp/initial_temp)**(i/n_iterations)
            
            # Adaptive perturbation scale
            acceptance_rate = n_accepted / (i + 1)
            if acceptance_rate < 0.2:
                perturbation_scale *= 0.95  # Too few accepts, reduce step size
            elif acceptance_rate > 0.5:
                perturbation_scale *= 1.05  # Too many accepts, increase step size
            
            # MH step
            V_current, J_current, accepted = self.metropolis_hastings_step(
                V_current, J_current, temperature, perturbation_scale
            )
            
            if accepted:
                n_accepted += 1
                
                # Update best
                if J_current > J_best:
                    V_best = V_current.copy()
                    J_best = J_current
            
            # Save history
            if i % save_every == 0:
                self.V_history.append(np.array(V_current))
                self.J_history.append(J_current)
                self.acceptance_history.append(acceptance_rate)
                self.temperature_history.append(temperature)
            
            # Progress report
            if (i+1) % 500 == 0:
                print(f"Iteration {i+1}/{n_iterations}:")
                print(f"  Current J/P₀ = {J_current:.6f}")
                print(f"  Best J/P₀ = {J_best:.6f}")
                print(f"  Acceptance rate = {acceptance_rate:.3f}")
                print(f"  Temperature = {temperature:.4f}")
                print(f"  Perturbation scale = {perturbation_scale:.4f}")
                print(f"  V mean = {np.mean(V_current):.4f}, std = {np.std(V_current):.4f}")
                print("-"*70)
        
        print("="*70)
        print(f"MCMC Complete!")
        print(f"Final acceptance rate: {n_accepted/n_iterations:.3f}")
        print(f"Best J/P₀ found: {J_best:.6f}")
        
        # Calculate improvement from initial
        if abs(self.J_history[0]) > 1e-10:
            improvement = ((J_best - self.J_history[0]) / abs(self.J_history[0])) * 100
            print(f"Improvement from V=0: {improvement:.1f}%")
        else:
            print(f"Starting from exact zero: J/P₀ went from undefined to {J_best:.6f}")
        
        print(f"\nSymmetry breaking achieved!")
        print(f"Final potential: mean={np.mean(V_best):.4f}, std={np.std(V_best):.4f}")
        print("="*70)
        
        return V_best, J_best
    
    def plot_mcmc_results(self, V_init, V_best):
        """
        Comprehensive visualization of MCMC results
        Emphasizes the symmetry breaking from V=0
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Plot 1: J/P_0 trace
        ax1 = fig.add_subplot(gs[0, :2])
        iterations_plot = np.arange(len(self.J_history)) * 10
        ax1.plot(iterations_plot, self.J_history, 'b-', linewidth=1.5, alpha=0.7)
        
        # Running maximum
        running_max = np.maximum.accumulate(self.J_history)
        ax1.plot(iterations_plot, running_max, 'r-', linewidth=2, 
                label='Best so far', alpha=0.8)
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('J/P₀ (Current)', fontsize=12)
        ax1.set_title('MCMC Trace: Spontaneous Symmetry Breaking from V=0', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Highlight the initial jump from V=0
        if len(self.J_history) > 1:
            ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.5,
                       label='First perturbation')
        
        # Plot 2: Acceptance rate
        ax2 = fig.add_subplot(gs[0, 2])
        iterations_acc = np.arange(len(self.acceptance_history)) * 10
        ax2.plot(iterations_acc, self.acceptance_history, 'g-', linewidth=2)
        ax2.axhline(y=0.234, color='r', linestyle='--', alpha=0.5, 
                   label='Optimal (~0.234)')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Acceptance Rate', fontsize=11)
        ax2.set_title('Acceptance Rate', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Temperature schedule
        ax3 = fig.add_subplot(gs[1, 0])
        iterations_temp = np.arange(len(self.temperature_history)) * 10
        ax3.semilogy(iterations_temp, self.temperature_history, 'orange', linewidth=2)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Temperature (log scale)', fontsize=11)
        ax3.set_title('Simulated Annealing\n(Higher T for V=0 start)', 
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, which='both')
        
        # Plot 4: Potential evolution
        ax4 = fig.add_subplot(gs[1, 1:])
        n_snapshots = min(len(self.V_history), 15)
        indices = np.linspace(0, len(self.V_history)-1, n_snapshots, dtype=int)
        cmap = plt.cm.viridis
        
        for idx, i in enumerate(indices):
            color = cmap(idx / (len(indices) - 1))
            iter_num = i * 10
            alpha_val = 0.3 + 0.7 * (idx / (len(indices) - 1))
            linewidth = 1 + 2 * (idx / (len(indices) - 1))
            if i == 0:
                label = f'V=0 (initial)'
            elif idx % 3 == 0:
                label = f'Iter {iter_num}'
            else:
                label = None
            ax4.plot(self.x, self.V_history[i], color=color, 
                    alpha=alpha_val, linewidth=linewidth, label=label)
        
        ax4.set_xlabel('Position x', fontsize=12)
        ax4.set_ylabel('V(x)', fontsize=12)
        ax4.set_title('Emergence of Structure: V(x) Evolution from Zero', 
                     fontsize=14, fontweight='bold')
        if n_snapshots <= 5:
            ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle=':', alpha=0.3, linewidth=2)
        
        # Plot 5: Initial (V=0) vs Best
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=3, alpha=0.5,
                   label='Initial V(x) = 0')
        ax5.plot(self.x, V_best, 'r-', label='Best V(x)', linewidth=3)
        ax5.fill_between(self.x, 0, V_best, alpha=0.2, color='red')
        ax5.set_xlabel('Position x', fontsize=11)
        ax5.set_ylabel('V(x)', fontsize=11)
        ax5.set_title('Zero Initial vs Best Potential\n(Symmetry Breaking)', 
                     fontsize=13, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Derivatives (showing asymmetry)
        ax6 = fig.add_subplot(gs[2, 1])
        dV_init = self.compute_derivative(jnp.array(V_init))  # Will be ~0
        dV_best = self.compute_derivative(jnp.array(V_best))
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=3, alpha=0.5,
                   label='Initial ∂V/∂x = 0')
        ax6.plot(self.x, dV_best, 'r-', label='Best ∂V/∂x', linewidth=2.5)
        ax6.fill_between(self.x, 0, dV_best, alpha=0.2, color='red')
        ax6.set_xlabel('Position x', fontsize=11)
        ax6.set_ylabel('∂V/∂x', fontsize=11)
        ax6.set_title('Gradient: Breaking Translation Symmetry', 
                     fontsize=13, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Statistics
        ax7 = fig.add_subplot(gs[2, 2])
        J_init = self.J_history[0]
        J_best = max(self.J_history)
        
        stats_text = f"""MCMC Statistics:

*** STARTING POINT ***
V(x) = 0 EVERYWHERE
(Exact zero, perfect symmetry)

Total iterations: {len(self.J_history)*10}
Samples saved: {len(self.J_history)}

Initial J/P₀: {J_init:.6f}
Best J/P₀: {J_best:.6f}
Final J/P₀: {self.J_history[-1]:.6f}

Symmetry Breaking:
V_best mean: {np.mean(V_best):.4f}
V_best std: {np.std(V_best):.4f}

Final acceptance: {self.acceptance_history[-1]:.3f}
Final temp: {self.temperature_history[-1]:.4f}

Parameters:
L = {self.L}
α = {self.alpha}
Grid = {self.n_points} points
"""
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
        ax7.axis('off')
        
        fig.suptitle('Spontaneous Symmetry Breaking: From V=0 to Asymmetric Ratchet',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current/mcmc_from_exact_zero_results.png', 
                   dpi=300, bbox_inches='tight')
        print("\nPlot saved: mcmc_from_exact_zero_results.png")
        return fig
    
    def plot_convergence_diagnostics(self):
        """
        Additional convergence diagnostics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = np.arange(len(self.J_history)) * 10
        
        # Trace plot with quantiles
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.J_history, 'b-', linewidth=1, alpha=0.5)
        
        # Running statistics
        window = 50
        if len(self.J_history) > window:
            running_mean = np.convolve(self.J_history, 
                                      np.ones(window)/window, mode='valid')
            ax1.plot(iterations[window-1:], running_mean, 'r-', 
                    linewidth=2, label=f'{window}-iteration moving avg')
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('J/P₀', fontsize=12)
        ax1.set_title('Trace: Breaking Away from V=0', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Histogram of J values
        ax2 = axes[0, 1]
        ax2.hist(self.J_history, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=np.mean(self.J_history), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(self.J_history):.4f}')
        ax2.axvline(x=np.max(self.J_history), color='g', linestyle='--', 
                   linewidth=2, label=f'Max: {np.max(self.J_history):.4f}')
        ax2.set_xlabel('J/P₀', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Sampled J/P₀', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Autocorrelation of J
        ax3 = axes[1, 0]
        max_lag = min(200, len(self.J_history)//2)
        autocorr = np.correlate(self.J_history - np.mean(self.J_history),
                               self.J_history - np.mean(self.J_history), 
                               mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        ax3.plot(np.arange(max_lag), autocorr[:max_lag], 'b-', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Lag', fontsize=12)
        ax3.set_ylabel('Autocorrelation', fontsize=12)
        ax3.set_title('Autocorrelation Function', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Energy landscape exploration
        ax4 = axes[1, 1]
        # Plot J vs iteration with color by temperature
        min_len = min(len(iterations), len(self.J_history), len(self.temperature_history))
        scatter = ax4.scatter(iterations[:min_len], 
                            self.J_history[:min_len], 
                            c=self.temperature_history[:min_len], 
                            cmap='hot', s=20, alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Temperature', fontsize=11)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('J/P₀', fontsize=12)
        ax4.set_title('Exploration (colored by temperature)', 
                     fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current/mcmc_from_exact_zero_diagnostics.png', 
                   dpi=300, bbox_inches='tight')
        print("Diagnostics saved: mcmc_from_exact_zero_diagnostics.png")
        return fig


def main():
    """Run MCMC optimization starting from EXACTLY V=0"""
    
    print("="*80)
    print("METROPOLIS-HASTINGS MCMC OPTIMIZATION")
    print("Variational Current J/P₀")
    print("="*80)
    print("\n" + "*"*80)
    print("*** STARTING FROM V = 0 EXACTLY (TRULY FLAT, ZERO EVERYWHERE) ***")
    print("*"*80)
    print("\nThis is the ultimate test of spontaneous symmetry breaking!")
    print("The system starts with PERFECT symmetry and NO structure.")
    print("Random perturbations will cause it to spontaneously develop asymmetry.")
    print("="*80)
    
    # Initialize
    L = 1.0
    alpha = 1.0
    n_points = 30  # Keep small for speed
    
    print(f"\nParameters:")
    print(f"  System length L = {L}")
    print(f"  Alpha parameter α = {alpha}")
    print(f"  Grid points = {n_points}")
    
    optimizer = MCMCVariationalOptimizer(L=L, alpha=alpha, n_points=n_points)
    
    # Initial potential: EXACTLY ZERO
    V_init = np.zeros(n_points)
    
    print(f"\nInitial potential: V(x) = 0 EXACTLY for all x")
    print("  - Perfect translational symmetry")
    print("  - Perfect reflection symmetry")
    print("  - No spatial structure whatsoever")
    print("\nThe MCMC will break this symmetry to maximize current!")
    
    # Run MCMC with higher initial temperature for V=0 start
    V_best, J_best = optimizer.run_mcmc(
        n_iterations=5000,
        V_init=V_init,
        initial_temp=2.0,  # Higher temp to escape V=0
        final_temp=0.01,
        perturbation_scale=0.15,  # Larger perturbations initially
        save_every=10
    )
    
    # Plot results
    print("\nGenerating visualizations...")
    optimizer.plot_mcmc_results(V_init, V_best)
    
    print("\nGenerating convergence diagnostics...")
    optimizer.plot_convergence_diagnostics()
    
    # Print final analysis
    print("\n" + "="*80)
    print("SYMMETRY BREAKING ANALYSIS")
    print("="*80)
    print(f"Initial state: V = 0 (completely flat)")
    print(f"Final state: V has structure!")
    print(f"  Mean value: {np.mean(V_best):.4f}")
    print(f"  Std dev: {np.std(V_best):.4f}")
    print(f"  Min value: {np.min(V_best):.4f}")
    print(f"  Max value: {np.max(V_best):.4f}")
    print(f"  Range: {np.max(V_best) - np.min(V_best):.4f}")
    print(f"\nCurrent achieved: J/P₀ = {J_best:.6f}")
    print("="*80)
    
    plt.show()
    
    return optimizer, V_init, V_best, J_best


if __name__ == "__main__":
    optimizer, V_init, V_best, J_best = main()
