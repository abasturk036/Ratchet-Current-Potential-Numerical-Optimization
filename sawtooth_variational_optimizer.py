"""
Variational optimization using FULL J/P_0 functional form
with SAWTOOTH RATCHET POTENTIAL (Angelani-style asymmetry)
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

class SawtoothVariationalOptimizer:
    """
    Variational analysis with SAWTOOTH RATCHET POTENTIAL
    
    V(x) = U0 * sawtooth(x, a)
    
    where a is the asymmetry parameter:
    - a = 0.5: symmetric
    - a ≠ 0.5: asymmetric (generates ratchet effect)
    
    Optimizes the asymmetry parameter 'a' to maximize J/P_0
    """
    
    def __init__(self, L=1.0, alpha=1.0, U0=2.0, n_points=50, n_periods=3):
        """
        Parameters:
        -----------
        L : float
            System length
        alpha : float
            Model parameter from variational equations
        U0 : float
            Potential barrier height
        n_points : int
            Number of spatial discretization points
        n_periods : int
            Number of sawtooth periods in domain
        """
        self.L = L
        self.alpha = alpha
        self.U0 = U0
        self.n_points = n_points
        self.n_periods = n_periods
        self.period = L / n_periods
        self.x = jnp.linspace(0, L, n_points)
        self.dx = L / (n_points - 1)
        
        # Trajectory tracking
        self.asymmetry_history = []
        self.objective_history = []
        self.V_history = []
        
    def sawtooth_potential(self, x, a):
        """
        Asymmetric sawtooth ratchet potential (Angelani-style)
        
        Parameters:
        -----------
        x : array
            Position
        a : float
            Asymmetry parameter (0 < a < 1)
            a = 0.5: symmetric
            a < 0.5: bias to left
            a > 0.5: bias to right
        
        Returns:
        --------
        V : array
            Potential energy at position x
        """
        # Position within period
        x_mod = jnp.mod(x, self.period) / self.period
        
        # Piecewise linear sawtooth
        V = jnp.where(x_mod < a,
                      self.U0 * x_mod / a,           # Rising part (0 to a)
                      self.U0 * (1 - x_mod) / (1 - a))  # Falling part (a to 1)
        return V
    
    def construct_V_from_asymmetry(self, a):
        """
        Construct full potential V(x) from asymmetry parameter a
        """
        return self.sawtooth_potential(self.x, a)
    
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
        # Add regularization for numerical stability
        denominator = V**2 - dV**2
        denominator = jnp.where(jnp.abs(denominator) < 1e-8, 
                                jnp.sign(denominator) * 1e-8, 
                                denominator)
        
        integrand = self.alpha * dV / denominator
        
        # Integrate from a to b
        integral = jnp.trapezoid(integrand[a_idx:b_idx+1], self.x[a_idx:b_idx+1])
        return jnp.exp(jnp.clip(integral, -50, 50))  # Clip to prevent overflow
    
    def compute_full_current(self, V):
        """
        Compute J/P_0 using the FULL functional form:
        
        J/P_0 = L(1 - ξ_{0,L}) / [TERM1 + TERM2]
        
        where:
        TERM1 = ∫₀^L dx [(∂_x V - α)/V²] ξ_{0,x} ∫₀^L dx' [V/(V² - (∂_x' V)²)] ξ_{x',x}
        
        TERM2 = (1 - ξ_{0,L}) ∫₀^L dx [1/(V² - (∂_x V)²)] ∫₀^x dx' [(∂²_x' V - α) ξ_{x',x}]
        """
        dV = self.compute_derivative(V)
        d2V = self.compute_second_derivative(V)
        
        # Numerator: L(1 - ξ_{0,L})
        xi_0L = self.compute_xi(V, 0, self.n_points-1)
        numerator = self.L * (1.0 - xi_0L)
        
        # TERM 1: Double integral term
        term1_outer_integrand = jnp.zeros(self.n_points)
        
        for i in range(self.n_points):
            xi_0x = self.compute_xi(V, 0, i)
            
            # Inner integral: ∫₀^L dx' [V/(V² - (∂_x' V)²)] ξ_{x',x}
            inner_integrand = jnp.zeros(self.n_points)
            for j in range(self.n_points):
                xi_xprime_x = self.compute_xi(V, j, i)
                denominator_inner = V[j]**2 - dV[j]**2
                denominator_inner = jnp.where(jnp.abs(denominator_inner) < 1e-8,
                                             jnp.sign(denominator_inner) * 1e-8,
                                             denominator_inner)
                inner_integrand = inner_integrand.at[j].set(
                    V[j] / denominator_inner * xi_xprime_x
                )
            
            inner_integral = jnp.trapezoid(inner_integrand, self.x)
            
            # Outer integrand factor
            V_safe = jnp.where(jnp.abs(V[i]) < 1e-8, 1e-8, V[i])
            outer_factor = (dV[i] - self.alpha) / (V_safe**2)
            
            term1_outer_integrand = term1_outer_integrand.at[i].set(
                outer_factor * xi_0x * inner_integral
            )
        
        term1 = jnp.trapezoid(term1_outer_integrand, self.x)
        
        # TERM 2: (1 - ξ_{0,L}) × ∫₀^L dx [stuff]
        term2_outer_integrand = jnp.zeros(self.n_points)
        
        for i in range(self.n_points):
            # Inner integral: ∫₀^x dx' [(∂²_x' V - α) ξ_{x',x}]
            inner_integrand = jnp.zeros(i+1)
            for j in range(i+1):
                xi_xprime_x = self.compute_xi(V, j, i)
                inner_integrand = inner_integrand.at[j].set(
                    (d2V[j] - self.alpha) * xi_xprime_x
                )
            
            if i > 0:
                inner_integral = jnp.trapezoid(inner_integrand, self.x[:i+1])
            else:
                inner_integral = 0.0
            
            # Outer integrand factor: 1/(V² - (∂_x V)²)
            denominator_outer = V[i]**2 - dV[i]**2
            denominator_outer = jnp.where(jnp.abs(denominator_outer) < 1e-8,
                                         jnp.sign(denominator_outer) * 1e-8,
                                         denominator_outer)
            outer_factor = 1.0 / denominator_outer
            
            term2_outer_integrand = term2_outer_integrand.at[i].set(
                outer_factor * inner_integral
            )
        
        term2_integral = jnp.trapezoid(term2_outer_integrand, self.x)
        term2 = (1.0 - xi_0L) * term2_integral
        
        # Full denominator
        denominator = term1 + term2
        
        # Add small regularization
        denominator = jnp.where(jnp.abs(denominator) < 1e-10,
                               jnp.sign(denominator) * 1e-10,
                               denominator)
        
        # Final current
        J_over_P0 = numerator / denominator
        
        return J_over_P0
    
    def objective_from_asymmetry(self, a):
        """
        Objective function: maximize J/P_0 by optimizing asymmetry parameter a
        """
        # Ensure a is in valid range
        a = jnp.clip(a, 0.1, 0.9)
        
        # Construct potential from asymmetry
        V = self.construct_V_from_asymmetry(a)
        
        # Compute current
        current = self.compute_full_current(V)
        
        # Return negative for minimization
        return -current
    
    def callback_function(self, a_current):
        """Callback to store optimization trajectory"""
        a_val = float(np.clip(a_current, 0.1, 0.9))
        V_current = self.construct_V_from_asymmetry(a_val)
        
        self.asymmetry_history.append(a_val)
        self.V_history.append(np.array(V_current))
        
        try:
            obj_val = float(self.objective_from_asymmetry(a_val))
            self.objective_history.append(-obj_val)  # Store actual J/P_0 value
        except:
            self.objective_history.append(np.nan)
    
    def optimize_asymmetry(self, a_init=0.5, method='L-BFGS-B'):
        """
        Optimize the asymmetry parameter to maximize J/P_0
        
        Parameters:
        -----------
        a_init : float
            Initial asymmetry (0.5 = symmetric)
        method : str
            Optimization method
        
        Returns:
        --------
        result : OptimizeResult
            Optimization result
        """
        # Reset history
        self.asymmetry_history = []
        self.objective_history = []
        self.V_history = []
        
        # Store initial point
        self.callback_function(a_init)
        
        # Gradient function using JAX
        grad_fn = jit(grad(self.objective_from_asymmetry))
        
        def objective_np(a):
            try:
                return float(self.objective_from_asymmetry(float(a)))
            except:
                return 1e10
        
        def grad_np(a):
            try:
                return float(grad_fn(float(a)))
            except:
                return 0.0
        
        def callback_wrapper(a):
            self.callback_function(float(a))
        
        # Optimize with bounds
        bounds = [(0.1, 0.9)]
        
        result = minimize(
            objective_np,
            a_init,
            method=method,
            jac=grad_np,
            callback=callback_wrapper,
            bounds=bounds,
            options={'maxiter': 100, 'disp': True, 'ftol': 1e-8}
        )
        
        return result
    
    def sweep_asymmetry(self, a_values=None):
        """
        Sweep through asymmetry values to map out J/P_0(a)
        """
        if a_values is None:
            a_values = np.linspace(0.1, 0.9, 30)
        
        J_values = []
        
        print(f"\nSweeping asymmetry parameter from {a_values[0]:.2f} to {a_values[-1]:.2f}...")
        
        for i, a in enumerate(a_values):
            if (i+1) % 5 == 0:
                print(f"  Progress: {i+1}/{len(a_values)}")
            
            try:
                J = -self.objective_from_asymmetry(a)
                J_values.append(J)
            except:
                J_values.append(np.nan)
        
        return a_values, np.array(J_values)
    
    def plot_results(self, a_init, a_opt, result):
        """
        Comprehensive visualization
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        V_init = self.construct_V_from_asymmetry(a_init)
        V_opt = self.construct_V_from_asymmetry(a_opt)
        
        # Plot 1: Sawtooth potential shapes
        ax1 = fig.add_subplot(gs[0, :])
        x_plot = np.linspace(0, self.L, 500)
        V_sym = self.sawtooth_potential(x_plot, 0.5)
        V_init_plot = self.sawtooth_potential(x_plot, a_init)
        V_opt_plot = self.sawtooth_potential(x_plot, a_opt)
        
        ax1.plot(x_plot, V_sym, 'k:', label=f'Symmetric (a=0.5)', linewidth=2, alpha=0.5)
        ax1.plot(x_plot, V_init_plot, 'b--', label=f'Initial (a={a_init:.3f})', linewidth=2.5)
        ax1.plot(x_plot, V_opt_plot, 'r-', label=f'Optimized (a={a_opt:.3f})', linewidth=3)
        
        # Mark asymmetry points
        for i in range(self.n_periods):
            x_asym_init = (i + a_init) * self.period
            x_asym_opt = (i + a_opt) * self.period
            ax1.axvline(x=x_asym_init, color='blue', linestyle=':', alpha=0.3)
            ax1.axvline(x=x_asym_opt, color='red', linestyle=':', alpha=0.3)
        
        ax1.set_xlabel('Position x', fontsize=14)
        ax1.set_ylabel('V(x)', fontsize=14)
        ax1.set_title('Sawtooth Ratchet Potential (Angelani-style)', 
                     fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, self.L)
        
        # Plot 2: Asymmetry parameter evolution
        ax2 = fig.add_subplot(gs[1, 0])
        if len(self.asymmetry_history) > 0:
            iterations = np.arange(len(self.asymmetry_history))
            ax2.plot(iterations, self.asymmetry_history, 'o-', 
                    linewidth=2, markersize=6, color='purple')
            ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Symmetric')
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Asymmetry Parameter a', fontsize=12)
            ax2.set_title('Evolution of Asymmetry', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        # Plot 3: J/P_0 evolution
        ax3 = fig.add_subplot(gs[1, 1])
        if len(self.objective_history) > 0:
            iterations = np.arange(len(self.objective_history))
            valid_mask = ~np.isnan(self.objective_history)
            ax3.plot(iterations[valid_mask], 
                    np.array(self.objective_history)[valid_mask],
                    's-', linewidth=2, markersize=6, color='darkgreen')
            ax3.set_xlabel('Iteration', fontsize=12)
            ax3.set_ylabel('J/P₀ (Current)', fontsize=12)
            ax3.set_title('Current vs Iteration', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Asymmetry sweep
        ax4 = fig.add_subplot(gs[1, 2])
        print("\nPerforming asymmetry sweep for visualization...")
        a_sweep, J_sweep = self.sweep_asymmetry()
        valid_mask = ~np.isnan(J_sweep)
        ax4.plot(a_sweep[valid_mask], J_sweep[valid_mask], 'o-', 
                linewidth=2, markersize=4, color='darkblue')
        ax4.axvline(x=0.5, color='k', linestyle='--', alpha=0.3, label='Symmetric')
        ax4.axvline(x=a_opt, color='r', linestyle='-', alpha=0.5, 
                   linewidth=2, label=f'Optimum: a={a_opt:.3f}')
        ax4.set_xlabel('Asymmetry Parameter a', fontsize=12)
        ax4.set_ylabel('J/P₀ (Current)', fontsize=12)
        ax4.set_title('Current vs Asymmetry', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Derivatives comparison
        ax5 = fig.add_subplot(gs[2, 0])
        dV_init = self.compute_derivative(V_init)
        dV_opt = self.compute_derivative(V_opt)
        ax5.plot(self.x, dV_init, 'b--', label='Initial ∂V/∂x', linewidth=2.5)
        ax5.plot(self.x, dV_opt, 'r-', label='Optimized ∂V/∂x', linewidth=2.5)
        ax5.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax5.set_xlabel('Position x', fontsize=12)
        ax5.set_ylabel('∂V/∂x', fontsize=12)
        ax5.set_title('First Derivative', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Second derivative
        ax6 = fig.add_subplot(gs[2, 1])
        d2V_opt = self.compute_second_derivative(V_opt)
        ax6.plot(self.x, d2V_opt, 'g-', linewidth=2.5)
        ax6.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax6.axhline(y=self.alpha, color='r', linestyle='--', alpha=0.5,
                   label=f'α = {self.alpha}')
        ax6.set_xlabel('Position x', fontsize=12)
        ax6.set_ylabel('∂²V/∂x²', fontsize=12)
        ax6.set_title('Second Derivative (Optimized)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Current comparison bar chart
        ax7 = fig.add_subplot(gs[2, 2])
        J_init = -self.objective_from_asymmetry(a_init)
        J_opt = -self.objective_from_asymmetry(a_opt)
        ax7.bar(['Initial\n(a={:.2f})'.format(a_init), 
                'Optimized\n(a={:.2f})'.format(a_opt)], 
               [J_init, J_opt],
               color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
        ax7.set_ylabel('J/P₀', fontsize=12)
        ax7.set_title('Current Comparison', fontsize=14, fontweight='bold')
        ax7.grid(True, axis='y', alpha=0.3)
        
        for i, (label, val) in enumerate([('Initial', J_init), ('Optimized', J_opt)]):
            if not np.isnan(val):
                ax7.text(i, val, f'{val:.4f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        
        # Plot 8: Potential evolution trajectory
        ax8 = fig.add_subplot(gs[3, :2])
        if len(self.V_history) > 0:
            n_snapshots = min(len(self.V_history), 8)
            indices = np.linspace(0, len(self.V_history)-1, n_snapshots, dtype=int)
            cmap = plt.cm.plasma
            
            for idx, i in enumerate(indices):
                color = cmap(idx / (len(indices)))
                a_val = self.asymmetry_history[i]
                label = f'a={a_val:.3f}'
                alpha_val = 0.3 + 0.7 * (idx / (len(indices) - 1))
                linewidth = 1 + 2 * (idx / (len(indices) - 1))
                ax8.plot(self.x, self.V_history[i], color=color, label=label,
                        alpha=alpha_val, linewidth=linewidth)
        
        ax8.set_xlabel('Position x', fontsize=12)
        ax8.set_ylabel('V(x)', fontsize=12)
        ax8.set_title('Evolution of Sawtooth Shape During Optimization', 
                     fontsize=14, fontweight='bold')
        ax8.legend(fontsize=9, ncol=4, loc='upper right')
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Statistics panel
        ax9 = fig.add_subplot(gs[3, 2])
        stats_text = f"""Optimization Results:

Initial asymmetry: {a_init:.4f}
Optimal asymmetry: {a_opt:.4f}
Shift from symmetric: {(a_opt-0.5):.4f}

Initial J/P₀: {J_init:.6f}
Final J/P₀: {J_opt:.6f}
Improvement: {((J_opt-J_init)/abs(J_init)*100 if abs(J_init)>1e-10 else 0):.1f}%

Iterations: {result.nit}
Success: {result.success}

Parameters:
L = {self.L}
α = {self.alpha}
U₀ = {self.U0}
Periods = {self.n_periods}
Grid = {self.n_points} points
"""
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        ax9.axis('off')
        
        fig.suptitle('Sawtooth Ratchet Optimization: Variational Current Analysis',
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.savefig('/home/claude/sawtooth_optimization_results.png', 
                   dpi=300, bbox_inches='tight')
        print("\nPlot saved: sawtooth_optimization_results.png")
        return fig


def main():
    """Run sawtooth ratchet optimization"""
    
    print("="*80)
    print("SAWTOOTH RATCHET CURRENT OPTIMIZATION")
    print("Asymmetry Parameter (Angelani-style)")
    print("="*80)
    print("\nOptimizing asymmetry 'a' of sawtooth potential to maximize J/P₀")
    print("  a = 0.5: symmetric (no ratchet effect)")
    print("  a ≠ 0.5: asymmetric (produces directed current)")
    print("="*80)
    
    # Initialize optimizer
    L = 3.0  # Length to fit multiple periods
    alpha = 1.0
    U0 = 2.0
    n_points = 50
    n_periods = 3
    
    print(f"\nParameters:")
    print(f"  System length L = {L}")
    print(f"  Alpha parameter α = {alpha}")
    print(f"  Barrier height U₀ = {U0}")
    print(f"  Number of periods = {n_periods}")
    print(f"  Grid points = {n_points}")
    
    optimizer = SawtoothVariationalOptimizer(
        L=L, alpha=alpha, U0=U0, n_points=n_points, n_periods=n_periods
    )
    
    # Start from symmetric case
    a_init = 0.7
    
    print(f"\nInitial asymmetry: a = {a_init} (symmetric)")
    print("Calculating initial J/P₀...")
    
    try:
        J_init = optimizer.objective_from_asymmetry(a_init)
        print(f"Initial J/P₀ = {J_init:.6f}")
    except Exception as e:
        print(f"Error: {e}")
        J_init = np.nan
    
    # Optimize
    print("\n" + "="*80)
    print("Starting optimization...")
    print("="*80)
    
    result = optimizer.optimize_asymmetry(a_init=a_init)
    
    a_opt = float(result.x)
    
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"Optimal asymmetry: a = {a_opt:.6f}")
    print(f"Optimized J/P₀ = {-result.fun:.6f}")
    print(f"Iterations: {result.nit}")
    print(f"Success: {result.success}")
    
    # Plot results
    print("\nGenerating visualizations...")
    optimizer.plot_results(a_init, a_opt, result)
    
    plt.show()
    
    return optimizer, a_init, a_opt, result


if __name__ == "__main__":
    optimizer, a_init, a_opt, result = main()
