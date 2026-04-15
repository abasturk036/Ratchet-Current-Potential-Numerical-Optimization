"""
Variational optimization using the FULL J/P_0 functional form
Based on the complete expression from the variational analysis document
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

class FullVariationalCurrentOptimizer:
    """
    Variational analysis using the COMPLETE J/P_0 functional form:
    
    J/P_0 = L(1 - ξ_{0,L}) / DENOMINATOR
    
    where DENOMINATOR has two terms:
    - First term: Double integral with (∂_x V - α)/V² and V/(V² - (∂_x V)²)
    - Second term: (1 - ξ_{0,L}) times integral with nested integral
    """
    
    def __init__(self, L=1.0, alpha=1.0, n_points=100):
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
        
        # Trajectory tracking
        self.trajectory = []
        self.objective_history = []
        
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
        # Outer integral: ∫₀^L dx [(∂_x V - α)/V²] ξ_{0,x} × [inner_integral]
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
        # Outer integral: ∫₀^L dx [1/(V² - (∂_x V)²)] × [inner_integral]
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
    
    def objective(self, V_params):
        """
        Objective function to maximize J/P_0
        We'll minimize the negative to find the maximum
        """
        V = V_params
        
        # Compute current using full formula
        current = self.compute_full_current(V)
        
        # Return negative for minimization
        return -current
    
    def callback_function(self, V_current):
        """Callback to store optimization trajectory"""
        self.trajectory.append(np.array(V_current).copy())
        try:
            obj_val = float(self.objective(jnp.array(V_current)))
            self.objective_history.append(-obj_val)  # Store actual J/P_0 value
        except:
            self.objective_history.append(np.nan)
    
    def optimize_potential(self, V_init=None, method='L-BFGS-B', track_trajectory=True):
        """
        Optimize the potential V(x) to extremize J/P_0
        
        Parameters:
        -----------
        V_init : array-like, optional
            Initial guess for V(x). If None, uses V(x) = 1 + 0.1*sin(2πx/L)
        method : str
            Optimization method for scipy.optimize.minimize
        track_trajectory : bool
            Whether to track the optimization trajectory
        
        Returns:
        --------
        result : OptimizeResult
            Optimization result containing optimized V(x)
        """
        if V_init is None:
            # Start from slightly perturbed constant
            V_init = 1.0 + 0.1 * jnp.sin(2 * jnp.pi * self.x / self.L)
        
        # Reset trajectory
        self.trajectory = []
        self.objective_history = []
        
        # Convert to numpy for scipy
        V_init_np = np.array(V_init)
        
        # Store initial point
        if track_trajectory:
            self.callback_function(V_init_np)
        
        # Gradient function using JAX
        grad_fn = jit(grad(self.objective))
        
        def objective_np(V):
            try:
                return float(self.objective(jnp.array(V)))
            except:
                return 1e10  # Return large value on error
        
        def grad_np(V):
            try:
                return np.array(grad_fn(jnp.array(V)))
            except:
                return np.zeros_like(V)  # Return zero gradient on error
        
        # Callback wrapper
        def callback_wrapper(V):
            if track_trajectory:
                self.callback_function(V)
        
        # Optimize with bounds to keep V positive
        bounds = [(0.1, 10.0)] * self.n_points
        
        result = minimize(
            objective_np,
            V_init_np,
            method=method,
            jac=grad_np,
            callback=callback_wrapper if track_trajectory else None,
            bounds=bounds,
            options={'maxiter': 500, 'disp': True, 'ftol': 1e-6}
        )
        
        return result
    
    def plot_full_results(self, V_init, V_opt, result):
        """
        Comprehensive visualization of optimization results
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Plot 1: Optimization trajectory of V(x)
        ax1 = fig.add_subplot(gs[0, :])
        if len(self.trajectory) > 0:
            n_snapshots = min(len(self.trajectory), 12)
            indices = np.linspace(0, len(self.trajectory)-1, n_snapshots, dtype=int)
            cmap = plt.cm.viridis
            
            for idx, i in enumerate(indices):
                color = cmap(idx / (len(indices) - 1))
                label = f'Iter {i}'
                alpha = 0.3 + 0.7 * (idx / (len(indices) - 1))
                linewidth = 1 + 2.5 * (idx / (len(indices) - 1))
                ax1.plot(self.x, self.trajectory[i], color=color, label=label,
                        alpha=alpha, linewidth=linewidth)
        
        ax1.set_xlabel('Position x', fontsize=14)
        ax1.set_ylabel('V(x)', fontsize=14)
        ax1.set_title('Evolution of Potential V(x) During Optimization (Full J/P₀ Formula)',
                     fontsize=16, fontweight='bold')
        ax1.legend(fontsize=9, ncol=3, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: J/P_0 over iterations
        ax2 = fig.add_subplot(gs[1, 0])
        iterations = np.arange(len(self.objective_history))
        valid_mask = ~np.isnan(self.objective_history)
        ax2.plot(iterations[valid_mask], np.array(self.objective_history)[valid_mask],
                'o-', linewidth=2, markersize=5, color='darkblue')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('J/P₀', fontsize=12)
        ax2.set_title('Current J/P₀ vs Iteration', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        if np.sum(valid_mask) > 1:
            initial_val = self.objective_history[0]
            final_val = self.objective_history[-1]
            if not np.isnan(initial_val) and not np.isnan(final_val):
                if abs(initial_val) > 1e-10:
                    improvement = ((final_val - initial_val) / abs(initial_val)) * 100
                    ax2.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
                            transform=ax2.transAxes, fontsize=11,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            verticalalignment='top', horizontalalignment='center')
                else:
                    ax2.text(0.5, 0.95, f'Initial: {initial_val:.6f}\nFinal: {final_val:.6f}',
                            transform=ax2.transAxes, fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            verticalalignment='top', horizontalalignment='center')
        
        # Plot 3: Initial vs Final V(x)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.x, V_init, 'b--', label='Initial V(x)', linewidth=3, alpha=0.7)
        ax3.plot(self.x, V_opt, 'r-', label='Optimized V(x)', linewidth=3)
        ax3.set_xlabel('Position x', fontsize=12)
        ax3.set_ylabel('V(x)', fontsize=12)
        ax3.set_title('Initial vs Optimized Potential', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Derivatives
        ax4 = fig.add_subplot(gs[1, 2])
        dV_init = self.compute_derivative(jnp.array(V_init))
        dV_opt = self.compute_derivative(jnp.array(V_opt))
        ax4.plot(self.x, dV_init, 'b--', label='Initial ∂V/∂x', linewidth=3, alpha=0.7)
        ax4.plot(self.x, dV_opt, 'r-', label='Optimized ∂V/∂x', linewidth=3)
        ax4.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax4.set_xlabel('Position x', fontsize=12)
        ax4.set_ylabel('∂V/∂x', fontsize=12)
        ax4.set_title('Spatial Derivative', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Current comparison
        ax5 = fig.add_subplot(gs[2, 0])
        J_init = -self.objective(jnp.array(V_init))
        J_opt = -self.objective(jnp.array(V_opt))
        ax5.bar(['Initial', 'Optimized'], [J_init, J_opt],
                color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
        ax5.set_ylabel('J/P₀', fontsize=12)
        ax5.set_title('Current Comparison (Full Formula)', fontsize=14, fontweight='bold')
        ax5.grid(True, axis='y', alpha=0.3)
        
        for i, (label, val) in enumerate([('Initial', J_init), ('Optimized', J_opt)]):
            if not np.isnan(val):
                ax5.text(i, val, f'{val:.4f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        
        # Plot 6: Second derivative
        ax6 = fig.add_subplot(gs[2, 1])
        d2V_opt = self.compute_second_derivative(jnp.array(V_opt))
        ax6.plot(self.x, d2V_opt, 'g-', linewidth=2.5)
        ax6.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax6.axhline(y=self.alpha, color='r', linestyle='--', alpha=0.5,
                   label=f'α = {self.alpha}')
        ax6.set_xlabel('Position x', fontsize=12)
        ax6.set_ylabel('∂²V/∂x²', fontsize=12)
        ax6.set_title('Second Derivative of Optimized V(x)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=11)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Optimization statistics
        ax7 = fig.add_subplot(gs[2, 2])
        stats_text = f"""Optimization Results:
        
Iterations: {result.nit}
Success: {result.success}
Message: {result.message[:30]}...

Initial J/P₀: {J_init:.6f}
Final J/P₀: {J_opt:.6f}

Parameters:
L = {self.L}
α = {self.alpha}
Grid points = {self.n_points}

Formula: FULL J/P₀
(with both double integral terms)
"""
        ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax7.axis('off')
        
        fig.suptitle('Variational Optimization with FULL J/P₀ Functional Form',
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current', dpi=300, bbox_inches='tight')
        print("\nPlot saved: full_optimization_results.png")
        return fig


def main():
    """Run the full variational analysis optimization"""
    
    print("="*80)
    print("Variational Analysis using FULL J/P₀ Functional Form")
    print("="*80)
    print("\nIncluding:")
    print("  - Double integral term with V/(V² - (∂V)²)")
    print("  - Nested integral term with ∂²V/∂x²")
    print("  - Complete denominator structure")
    print("="*80)
    
    # Initialize optimizer
    L = 1.0
    alpha = 1.0
    n_points = 50  # Reduce for computational speed
    
    print(f"\nParameters:")
    print(f"  System length L = {L}")
    print(f"  Alpha parameter = {alpha}")
    print(f"  Grid points = {n_points}")
    
    optimizer = FullVariationalCurrentOptimizer(L=L, alpha=alpha, n_points=n_points)
    
    # Initial potential: slightly perturbed
    V_init = 1.0 + 0.1 * np.sin(2 * np.pi * optimizer.x / L)
    
    print(f"\nInitial potential: V(x) = 1 + 0.1*sin(2πx/L)")
    print("Calculating initial J/P₀...")
    
    try:
        J_init = -optimizer.objective(jnp.array(V_init))
        print(f"Initial J/P₀ = {J_init:.6f}")
    except Exception as e:
        print(f"Error calculating initial J/P₀: {e}")
        J_init = np.nan
    
    # Optimize
    print("\n" + "="*80)
    print("Starting optimization with FULL functional form...")
    print("(This may take longer due to nested integrals)")
    print("="*80)
    
    result = optimizer.optimize_potential(V_init=V_init, method='L-BFGS-B',
                                         track_trajectory=True)
    
    V_opt = result.x
    
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"Optimized J/P₀ = {-result.fun:.6f}")
    print(f"Number of iterations: {result.nit}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Trajectory points captured: {len(optimizer.trajectory)}")
    
    # Plot results
    print("\nGenerating comprehensive visualization...")
    optimizer.plot_full_results(V_init, V_opt, result)
    
    plt.show()
    
    return optimizer, V_init, V_opt, result


if __name__ == "__main__":
    optimizer, V_init, V_opt, result = main()
