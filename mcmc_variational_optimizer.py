"""
Metropolis-Hastings MCMC Optimization of J/P_0
Starting from small potential with random perturbations
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
    
    Starts from small potential and explores via random perturbations
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
        """
        dV = self.compute_derivative(V)
        
        # Check for pathological V
        if jnp.any(V < 0.05) or jnp.any(V > 10.0):
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
        V_proposed = np.clip(V_proposed, 0.1, 5.0)
        
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
                 initial_temp=1.0, final_temp=0.01,
                 perturbation_scale=0.05, save_every=10):
        """
        Run MCMC optimization
        
        Parameters:
        -----------
        n_iterations : int
            Number of MCMC steps
        V_init : array, optional
            Initial potential (if None, starts small)
        initial_temp : float
            Starting temperature for simulated annealing
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
        # Initialize
        if V_init is None:
            # Start from small constant potential
            V_init = 0.5 * np.ones(self.n_points)
        
        V_current = jnp.array(V_init)
        
        # Evaluate initial current
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
        
        print(f"\nStarting MCMC with {n_iterations} iterations")
        print(f"Initial J/P₀ = {J_current:.6f}")
        print("="*60)
        
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
                print("-"*60)
        
        print("="*60)
        print(f"MCMC Complete!")
        print(f"Final acceptance rate: {n_accepted/n_iterations:.3f}")
        print(f"Best J/P₀ found: {J_best:.6f}")
        print(f"Improvement from initial: {((J_best-self.J_history[0])/abs(self.J_history[0])*100 if abs(self.J_history[0])>1e-10 else 0):.1f}%")
        
        return V_best, J_best
    
    def plot_mcmc_results(self, V_init, V_best):
        """
        Comprehensive visualization of MCMC results
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Plot 1: J/P_0 trace
        ax1 = fig.add_subplot(gs[0, :2])
        iterations_plot = np.arange(len(self.J_history)) * 10  # multiply by save_every
        ax1.plot(iterations_plot, self.J_history, 'b-', linewidth=1.5, alpha=0.7)
        
        # Running maximum
        running_max = np.maximum.accumulate(self.J_history)
        ax1.plot(iterations_plot, running_max, 'r-', linewidth=2, 
                label='Best so far', alpha=0.8)
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('J/P₀ (Current)', fontsize=12)
        ax1.set_title('MCMC Trace: Current vs Iteration', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Acceptance rate
        ax2 = fig.add_subplot(gs[0, 2])
        # Make sure dimensions match
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
        ax3.set_title('Simulated Annealing', fontsize=13, fontweight='bold')
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
            label = f'Iter {iter_num}' if idx % 3 == 0 else None
            ax4.plot(self.x, self.V_history[i], color=color, 
                    alpha=alpha_val, linewidth=linewidth, label=label)
        
        ax4.set_xlabel('Position x', fontsize=12)
        ax4.set_ylabel('V(x)', fontsize=12)
        ax4.set_title('Evolution of Potential V(x)', fontsize=14, fontweight='bold')
        if n_snapshots <= 5:
            ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Initial vs Best
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(self.x, V_init, 'b--', label='Initial V(x)', linewidth=3, alpha=0.7)
        ax5.plot(self.x, V_best, 'r-', label='Best V(x)', linewidth=3)
        ax5.set_xlabel('Position x', fontsize=11)
        ax5.set_ylabel('V(x)', fontsize=11)
        ax5.set_title('Initial vs Best Potential', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Derivatives
        ax6 = fig.add_subplot(gs[2, 1])
        dV_init = self.compute_derivative(jnp.array(V_init))
        dV_best = self.compute_derivative(jnp.array(V_best))
        ax6.plot(self.x, dV_init, 'b--', label='Initial ∂V/∂x', linewidth=2.5, alpha=0.7)
        ax6.plot(self.x, dV_best, 'r-', label='Best ∂V/∂x', linewidth=2.5)
        ax6.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax6.set_xlabel('Position x', fontsize=11)
        ax6.set_ylabel('∂V/∂x', fontsize=11)
        ax6.set_title('Spatial Derivative', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Statistics
        ax7 = fig.add_subplot(gs[2, 2])
        J_init = self.J_history[0]
        J_best = max(self.J_history)
        
        stats_text = f"""MCMC Statistics:

Total iterations: {len(self.J_history)*10}
Samples saved: {len(self.J_history)}

Initial J/P₀: {J_init:.6f}
Best J/P₀: {J_best:.6f}
Final J/P₀: {self.J_history[-1]:.6f}

Improvement: {((J_best-J_init)/abs(J_init)*100 if abs(J_init)>1e-10 else 0):.1f}%

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
        
        fig.suptitle('Metropolis-Hastings MCMC Optimization of J/P₀',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current', 
                   dpi=300, bbox_inches='tight')
        print("\nPlot saved: mcmc_optimization_results.png")
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
        ax1.set_title('Trace Plot with Moving Average', fontsize=13, fontweight='bold')
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
        # Make sure all arrays have same length
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
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current', dpi=300, bbox_inches='tight')
        print("Diagnostics saved: mcmc_diagnostics.png")
        return fig


def main():
    """Run MCMC optimization"""
    
    print("="*70)
    print("METROPOLIS-HASTINGS MCMC OPTIMIZATION")
    print("Variational Current J/P₀")
    print("="*70)
    print("\nStarting from small potential with random perturbations")
    print("Using simulated annealing for global exploration")
    print("="*70)
    
    # Initialize
    L = 1.0
    alpha = 1.0
    n_points = 30  # Keep small for speed
    
    print(f"\nParameters:")
    print(f"  System length L = {L}")
    print(f"  Alpha parameter α = {alpha}")
    print(f"  Grid points = {n_points}")
    
    optimizer = MCMCVariationalOptimizer(L=L, alpha=alpha, n_points=n_points)
    
    # Initial potential: small and smooth
    V_init = 0.5 + 0.1 * np.sin(2*np.pi*optimizer.x/L)
    
    print(f"\nInitial potential: V(x) = 0.5 + 0.1*sin(2πx/L)")
    
    # Run MCMC
    V_best, J_best = optimizer.run_mcmc(
        n_iterations=5000,
        V_init=V_init,
        initial_temp=1.0,
        final_temp=0.01,
        perturbation_scale=0.1,
        save_every=10
    )
    
    # Plot results
    print("\nGenerating visualizations...")
    optimizer.plot_mcmc_results(V_init, V_best)
    
    print("\nGenerating convergence diagnostics...")
    optimizer.plot_convergence_diagnostics()
    
    plt.show()
    
    return optimizer, V_init, V_best, J_best


if __name__ == "__main__":
    optimizer, V_init, V_best, J_best = main()
