"""
Parallel Tempering MCMC Optimization of J/P(0) written in collaboration with Claude Sonnet 4.5
Multiple HMMC (Hasting-Metropolis Markov Chain) chains at fixed temperatures run in parallel
Starting from V = 0 with gradient constraint: -1 < ∂V/∂x < 1

Version 5 (modified 3/3/2026)
- Fourier analysis of the final optimized potential V(x)
"""

import matplotlib.pyplot as plt
import numpy as np

class ParallelTemperingOptimizer:
    """
    Parallel HMMC for V(x) to maximize J/P_0. The algorithm runs 5 chains at different fixed temperatures
    Here we Enforces gradient constraint: -1 < ∂V/∂x < 1
    """
    
    def __init__(self, L=1.0, alpha=1.0, n_points=30, temperatures=[0.01, 0.1, 0.5, 1.0, 2.0]):
        """
        Parameters:
        -----------
        L : float, System length
        alpha : float, tumble rate
        n_points : int, the number of spatial discretization points
        temperatures : list, the fixed temperatures for parallel chains
        """
        self.L = L
        self.alpha = alpha
        self.n_points = n_points
        self.x = np.linspace(0, L, n_points)
        self.dx = L / (n_points - 1)
        self.temperatures = temperatures
        self.n_chains = len(temperatures)
        
        # MCMC tracking for each temperature
        self.V_history = {T: [] for T in temperatures}
        self.J_history = {T: [] for T in temperatures}
        self.acceptance_history = {T: [] for T in temperatures}
        
    def compute_derivative(self, V):
        """Compute spatial derivative using finite differences"""
        return np.gradient(V, self.dx)
    
    def check_gradient_constraint(self, V, epsilon):
        """
        Check if gradient constraint is satisfied: -1 < ∂V/∂x < 1. This avoids singularity in J/P_0 formula where 1 - (∂V)² appears in denominator
        """
        dV = self.compute_derivative(V)
        return np.all(dV > -1.0 + epsilon) and np.all(dV < 1.0 - epsilon)
    
    def compute_xi(self, V, a_idx, b_idx):
        """
        Compute ξ_{a,b} = exp[∫_a^b dc α∂_c V / (1 - (∂_c V)^2)]
        """
        if a_idx >= b_idx:
            return 1.0
            
        dV = self.compute_derivative(V)
        denominator = 1 - dV**2
        
        # With gradient constraint |dV| < 1, 
        denominator = np.where(np.abs(denominator) < 1e-8, 
                               np.sign(denominator) * 1e-8, 
                               denominator)
        
        integrand = self.alpha * dV / denominator
        integral = np.trapz(integrand[a_idx:b_idx+1], self.x[a_idx:b_idx+1])
        return np.exp(np.clip(integral, -50, 50))

    def compute_A_integrand(self, V):
        """
        Compute A(x) = [2f ∂ₓf + αf] / (1 - f²),  where f(x) = -∂ₓV(x)
        Note: with v=1, v² - f² = 1 - (∂V/∂x)²
        """
        dV = self.compute_derivative(V)
        f = -dV
        df = np.gradient(f, self.dx)
        denom = 1.0 - f**2
        denom = np.where(np.abs(denom) < 1e-8,
                         np.sign(denom + 1e-15) * 1e-8, denom)
        return (2.0 * f * df + self.alpha * f) / denom

    def compute_b_integrand(self, V):
        """
        Compute b(x) = [∂ₓf - α] / (1 - f²),  where f(x) = -∂ₓV(x)
        """
        dV = self.compute_derivative(V)
        f = -dV
        df = np.gradient(f, self.dx)
        denom = 1.0 - f**2
        denom = np.where(np.abs(denom) < 1e-8,
                         np.sign(denom + 1e-15) * 1e-8, denom)
        return (df - self.alpha) / denom

    def compute_cumulative_A(self, A):
        """
        Compute I(x) = ∫₀ˣ A(x') dx' for all x on the grid via cumulative trapezoid.
        Returns array of shape (n_points,) with I[0] = 0.
        """
        cum = np.concatenate([[0.0],
                              np.cumsum(0.5 * (A[:-1] + A[1:]) * self.dx)])
        return cum  # cum[i] ≈ ∫₀^{x_i} A dx'

    def compute_full_current(self, V):
        """
        Compute J/P(0) using the FULL Eq. 13 expression from the paper:

            J/P₀ = C / [ (1/L) ∫₀ᴸ exp(∫₀ˣ A dx') dx
                        + (C/L) ∫₀ᴸ ∫₀ˣ exp(∫ₓ'ˣ A dx'') b(x') dx' dx ]

        where C = J/P(0) is given by Eq. 10:
            C = [1 - exp(∫₀ᴸ A dx)] / [∫₀ᴸ b(x) exp(∫₀ˣ A dx') dx]

        With gradient constraint: -1 < ∂V/∂x < 1
        """
        epsilon = 0.1 #tune gradident constraint to avoid artificial ballooning of J/P_0

        # Check gradient constraint
        if not self.check_gradient_constraint(V, epsilon):
            return -1e10  # the step is rejected if constraint violated
        
        dV = self.compute_derivative(V)
        
        # Check for pathological V
        if np.all(np.abs(V) < 0.01):
            return -1e2  # Reject V≈0 everywhere
        
        if np.any(V < 0.1) or np.any(V > 10.0):
            return -1e2  # Keep V bounded

        A   = self.compute_A_integrand(V)
        b   = self.compute_b_integrand(V)
        cum = self.compute_cumulative_A(A)   # cum[i] = ∫₀^{x_i} A dx'

        # exp(∫₀ˣ A dx') at every grid point
        exp_cum = np.exp(np.clip(cum, -50.0, 50.0))

        # ---- Compute C (Eq. 10) ----
        xi_0L = exp_cum[-1]  # exp(∫₀ᴸ A dx)

        if not np.isfinite(xi_0L):
            return -1e10

        numerator_C = 1.0 - xi_0L

        denom_C_integrand = b * exp_cum  # b(x) exp(∫₀ˣ A dx')
        denominator_C = np.trapz(denom_C_integrand, self.x)
        if np.abs(denominator_C) < 1e-10:
            denominator_C = np.sign(denominator_C) * 1e-10

        if not np.isfinite(denominator_C):
            return -1e10

        C = numerator_C / denominator_C

        if not np.isfinite(C):
            return -1e10

        # ---- First term of Eq. 13 denominator ----
        # (1/L) ∫₀ᴸ exp(∫₀ˣ A dx') dx
        term1 = np.trapz(exp_cum, self.x) / self.L

        # ---- Second term of Eq. 13 denominator ----
        # (C/L) ∫₀ᴸ [ ∫₀ˣ exp(∫ₓ'ˣ A dx'') b(x') dx' ] dx
        #
        # Factor the inner integral to avoid O(N²) loop:
        #   ∫₀ˣ exp(∫ₓ'ˣ A dx'') b(x') dx'
        #   = exp(∫₀ˣ A dx'') · ∫₀ˣ [b(x') / exp(∫₀^{x'} A dx'')] dx'
        #   = exp_cum[x] · ∫₀ˣ (b / exp_cum) dx'

        b_over_exp = b / np.where(np.abs(exp_cum) < 1e-15, 1e-15, exp_cum)
        # Cumulative integral of b_over_exp from 0 to x_i 
        cum_b_over_exp = np.concatenate(
            [[0.0],
             np.cumsum(0.5 * (b_over_exp[:-1] + b_over_exp[1:]) * self.dx)])

        inner_integrals = exp_cum * cum_b_over_exp
        term2 = C * np.trapz(inner_integrals, self.x) / self.L

        denominator_JP0 = term1 + term2

        if not np.isfinite(denominator_JP0) or np.abs(denominator_JP0) < 1e-12:
            return -1e10

        result = C / denominator_JP0

        # Safeguard against explosion
        if not np.isfinite(result) or np.abs(result) > 1e6:
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
            
            # Generate V = 0 potential
            perturbation = np.zeros(self.n_points)
            
            # Add small noise
            perturbation += scale * 0.1 * np.random.randn(self.n_points)

            # Deep copy to avoid aliasing: always materialise a fresh numpy array
            V_proposed = np.array(V_current, copy=True) + perturbation
            
            # Keep V positive and bounded
            V_proposed = np.clip(V_proposed, 0.1, 5.0)
            
            # Check gradient constraint
            if self.check_gradient_constraint(V_proposed, 0.1):
                return V_proposed
        
        # If all attempts fail, return current V (rejection)
        return V_current
    
    def metropolis_step(self, V_current, J_current, temperature, perturbation_scale):
        """
        A single Metropolis-Hastings step at fixed temperature
        """
        # Propose new V
        V_proposed = self.propose_perturbation(V_current, perturbation_scale)
        
        # Check if proposal was rejected due to constraint
        if np.allclose(V_proposed, V_current):
            return V_current, J_current, False
        
        # Evaluate objective
        try:
            J_proposed = float(self.compute_full_current(V_proposed))
            
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
    
    def run_parallel_tempering(self, n_iterations=10000, V_init=None, 
                               perturbation_scale=0.1, save_every=10):
        """
        Run parallel markov chains with fixed temperatures Each chain runs independently at its own fixed temperature.
        """
        # Initialize all chains from same V_init
        if V_init is None:
            V_init = np.zeros(self.n_points)
        
        # State for each chain — each gets its own independent deep copy to prevent aliasing
        V_chains = {T: np.array(V_init, copy=True) for T in self.temperatures}
        J_chains = {T: -1e10 for T in self.temperatures}
        
        # Initialize J for each chain
        for T in self.temperatures:
            try:
                J_chains[T] = float(self.compute_full_current(V_chains[T]))
            except:
                J_chains[T] = -1e10
        
        # Best overall — deep copy to prevent aliasing
        V_best = np.array(V_chains[self.temperatures[0]], copy=True)
        J_best = J_chains[self.temperatures[0]]
        
        # Reset history
        for T in self.temperatures:
            self.V_history[T] = [np.array(V_chains[T], copy=True)]
            self.J_history[T] = [J_chains[T]]
            self.acceptance_history[T] = []
        
        n_accepted = {T: 0 for T in self.temperatures}
        
        print(f"\n" + "="*80)
        print(f"PARALLEL HMMC")
        print(f"Running {self.n_chains} chains at a fixed temperatures")
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
                    
                    # Update global best — deep copy to avoid aliasing
                    if J_chains[T] > J_best:
                        V_best = np.array(V_chains[T], copy=True)
                        J_best = J_chains[T]
            
            # Save history
            if i % save_every == 0:
                for T in self.temperatures:
                    self.V_history[T].append(np.array(V_chains[T], copy=True))
                    self.J_history[T].append(J_chains[T])
                    acceptance_rate = n_accepted[T] / (i + 1)
                    self.acceptance_history[T].append(acceptance_rate)
            
            # Progress report
            if (i+1) % 1000 == 0:
                print(f"\nIteration {i+1}/{n_iterations}:")
                for T in self.temperatures:
                    acc_rate = n_accepted[T] / (i + 1)
                    print(f"  T={T:.2e}: J={J_chains[T]:12.4f}, acceptance={acc_rate:.3f}")
                print(f"  Best overall: J={J_best:.4f}")
                print("-"*80)
        
        print("\n" + "="*80)
        print(f"Best J/P₀ found: {J_best:.6f}")
        print(f"\nFinal acceptance rates:")
        for T in self.temperatures:
            final_acc = n_accepted[T] / n_iterations
            print(f"  T={T:.2e}: {final_acc:.3f}")
        print("="*80)
        
        return V_best, J_best

    def tune_temperature(self, V_init=None, n_samples=500, perturbation_scale=0.1,
                         target_low=0.10, target_high=0.20):
        """
        Find a temperature s.t. acceptance rate is 10 - 20%.

        Strategy: collect a pilot sample of |delta_J| values from random walk
        proposals, then binary-search for the temperature T* such that the
        expected acceptance rate  E[min(1, exp(delta_J / T*))] falls in
        [target_low, target_high].

        Parameters:
        -----------
        V_init           : initial potential (zeros if None)
        n_samples        : number of pilot proposals to collect
        perturbation_scale : passed to propose_perturbation
        target_low/high  : desired acceptance rate band (default 10-20%)

        Returns:
        --------
        T_opt : float, recommended temperature
        """
        if V_init is None:
            V_init = np.zeros(self.n_points)

        V_current = np.array(V_init, copy=True)
        # Warm up to a non-pathological starting point
        for _ in range(50):
            V_prop = self.propose_perturbation(V_current, perturbation_scale)
            if float(self.compute_full_current(V_prop)) > -1e5:
                V_current = V_prop
                break

        # Collect pilot |delta_J| samples from accepted and rejected proposals
        delta_J_samples = []
        J_current = float(self.compute_full_current(V_current))

        print("\nTuning temperature — collecting pilot samples...")
        for _ in range(n_samples):
            V_prop = self.propose_perturbation(V_current, perturbation_scale)
            J_prop = float(self.compute_full_current(V_prop))
            if np.isfinite(J_prop) and J_prop > -1e5:
                delta_J_samples.append(J_prop - J_current)
                # Accept with prob 1 to keep exploring (pilot only)
                if J_prop > J_current:
                    V_current = V_prop
                    J_current = J_prop

        if len(delta_J_samples) < 10:
            print("  Warning: too few valid proposals — try increasing n_samples or perturbation_scale.")
            return None

        delta_J_arr = np.array(delta_J_samples)

        def expected_acceptance(T):
            """E[min(1, exp(dJ/T))] over the pilot sample"""
            return np.mean(np.minimum(1.0, np.exp(np.clip(delta_J_arr / T, -50, 50))))

        # Binary search over temperature
        T_lo, T_hi = 1e-6, 1e12
        for _ in range(100):
            T_mid = np.sqrt(T_lo * T_hi)  # geometric midpoint
            acc = expected_acceptance(T_mid)
            if acc < target_low:
                T_lo = T_mid   # too cold — increase T
            elif acc > target_high:
                T_hi = T_mid   # too hot — decrease T
            else:
                break

        T_opt = np.sqrt(T_lo * T_hi)
        acc_opt = expected_acceptance(T_opt)

        print(f"  Pilot delta_J stats:  mean={np.mean(delta_J_arr):.4f}, "
              f"std={np.std(delta_J_arr):.4f}, "
              f"median|dJ|={np.median(np.abs(delta_J_arr)):.4f}")
        print(f"  Target acceptance:    {target_low*100:.0f}% – {target_high*100:.0f}%")
        print(f"  Recommended T*:       {T_opt:.4e}  (predicted acceptance {acc_opt*100:.1f}%)")

        return T_opt

    def fourier_analysis(self, V, n_modes=20, label=''):
        """
        Perform Fourier analysis on the potential V(x) over one period [0, L].

        Computes the real DFT: V(x) = a_0/2 + Σ_k [ a_k cos(2πkx/L) + b_k sin(2πkx/L) ]

        Parameters:
        -----------
        V      : 1D numpy array, the potential over [0, L]
        n_modes: int, number of Fourier modes to display (beyond the DC component)
        label  : str, optional label for print output (e.g. temperature)

        Returns:
        --------
        freqs      : spatial frequencies [cycles per unit length]
        amplitudes : |c_k| = sqrt(a_k² + b_k²) for k = 0, 1, ..., n_modes
        phases     : phase angle φ_k = arctan2(b_k, a_k) in radians
        V_reconstructed : V reconstructed from the first n_modes Fourier modes
        """
        N = len(V)

        # Real FFT — output has N//2 + 1 complex coefficients
        coeffs = np.fft.rfft(V) / N   # normalise so |c_k| is amplitude in V units

        # Spatial frequencies: k/L cycles per unit length
        freqs = np.fft.rfftfreq(N, d=self.dx)   # cycles per unit length

        # Amplitudes and phases
        amplitudes = np.abs(coeffs)
        amplitudes[1:-1] *= 2       # double-count positive freqs (except DC and Nyquist)
        phases = np.angle(coeffs)   # radians

        # Reconstruct V from first n_modes modes (k = 0 … n_modes)
        coeffs_truncated = np.zeros_like(coeffs)
        coeffs_truncated[:n_modes + 1] = coeffs[:n_modes + 1]
        V_reconstructed = np.fft.irfft(coeffs_truncated * N, n=N)

        # Print dominant modes
        print(f"\nFourier analysis {label}")
        print(f"{'Mode k':>8}  {'Freq (1/L)':>12}  {'Amplitude':>12}  {'Phase (rad)':>12}")
        print("-" * 50)
        for k in range(min(n_modes + 1, len(freqs))):
            print(f"{k:>8d}  {freqs[k]:>12.4f}  {amplitudes[k]:>12.6f}  {phases[k]:>12.4f}")

        return freqs, amplitudes, phases, V_reconstructed

    def plot_fourier(self, V_best, n_modes=20):
        """
        Plot Fourier analysis of the best potential V_best:
          - Amplitude spectrum (bar chart of |c_k|)
          - Phase spectrum
          - V(x) vs Fourier reconstruction overlaid
        """
        freqs, amplitudes, phases, V_recon = self.fourier_analysis(
            V_best, n_modes=n_modes, label='(best potential)')

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('Fourier Analysis of Optimal Potential V(x)',
                     fontsize=15, fontweight='bold')

        k_vals = np.arange(len(freqs))

        # Plot 1: Amplitude spectrum
        ax1 = axes[0]
        ax1.bar(k_vals[:n_modes + 1], amplitudes[:n_modes + 1],
                color='steelblue', edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Mode k', fontsize=12)
        ax1.set_ylabel('Amplitude |cₖ|  [V units]', fontsize=12)
        ax1.set_title('Amplitude Spectrum', fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, axis='y')

        # Annotate dominant modes (top 5 by amplitude, excluding DC)
        top_idx = np.argsort(amplitudes[1:n_modes + 1])[::-1][:5] + 1
        for idx in top_idx:
            ax1.annotate(f'k={idx}', xy=(idx, amplitudes[idx]),
                         xytext=(idx + 0.3, amplitudes[idx] * 1.5),
                         fontsize=9, color='darkred')

        '''
        # Plot 2: Phase spectrum
        ax2 = axes[1]
        ax2.bar(k_vals[:n_modes + 1], phases[:n_modes + 1],
                color='darkorange', edgecolor='black', alpha=0.8)
        ax2.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('Mode k', fontsize=12)
        ax2.set_ylabel('Phase φₖ  [rad]', fontsize=12)
        ax2.set_title('Phase Spectrum', fontsize=13, fontweight='bold')
        ax2.set_ylim(-np.pi, np.pi)
        ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        ax2.grid(True, alpha=0.3, axis='y')
        '''

        # Plot 3: V(x) vs Fourier reconstruction
        ax3 = axes[2]
        ax3.plot(self.x, V_best, '-', color='steelblue', linewidth=2,
                 alpha=0.9, label='V(x) optimized')
        ax3.plot(self.x, V_recon, '--', color='darkred', linewidth=2,
                 alpha=0.85, label=f'Fourier reconstruction (k ≤ {n_modes})')
        ax3.set_xlabel('Position x', fontsize=12)
        ax3.set_ylabel('V(x)', fontsize=12)
        ax3.set_title('V(x) vs Fourier Reconstruction', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current/HMMC_V5_fourier.png',
                   dpi=300, bbox_inches='tight')
        print("\nFourier plot saved: HMMC_V5_fourier.png")
        return fig

    def plot_all_temperatures(self, V_init, V_best):
        """
        Plot results for all temperatures on same plots
        """
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)
        
        # Define colors for each temperature
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        temp_colors = {T: colors[i] for i, T in enumerate(self.temperatures)}
        
        # Plot 1: J/P_0 traces for all temperatures (full width)
        ax1 = fig.add_subplot(gs[0, :])
        for T in self.temperatures:
            iterations = np.arange(len(self.J_history[T])) * 10
            ax1.plot(iterations, self.J_history[T], '-', 
                    color=temp_colors[T], linewidth=2, alpha=0.7, label=f'T={T:.2e}')
        
        ax1.set_xlabel('Iteration', fontsize=13)
        ax1.set_ylabel('J/P₀ (Current)', fontsize=13)
        ax1.set_title('Current Evolution at All Temperatures (Fixed T per chain)', 
                     fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, ncol=5, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Running maximum for all temperatures (full width)
        ax2 = fig.add_subplot(gs[1, :])
        for T in self.temperatures:
            iterations = np.arange(len(self.J_history[T])) * 10
            running_max = np.maximum.accumulate(self.J_history[T])
            ax2.plot(iterations, running_max, '-', 
                    color=temp_colors[T], linewidth=2.5, alpha=0.8, label=f'T={T:.2e}')
        
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
                    color=temp_colors[T], linewidth=2, alpha=0.7, label=f'T={T:.2e}')
        
        ax3.axhline(y=0.234, color='k', linestyle='--', alpha=0.3, label='Optimal')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Acceptance Rate', fontsize=12)
        ax3.set_title('Acceptance Rates (All T)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # Plot 4: Histogram of final J values
        ax4 = fig.add_subplot(gs[2, 1])
        final_J_values = [self.J_history[T][-1] for T in self.temperatures]
        bars = ax4.bar(range(len(self.temperatures)), final_J_values,
                      color=[temp_colors[T] for T in self.temperatures],
                      alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_xticks(range(len(self.temperatures)))
        ax4.set_xticklabels([f'T={T:.2e}' for T in self.temperatures], fontsize=10)
        ax4.set_ylabel('Final J/P₀', fontsize=12)
        ax4.set_title('Final Current by Temperature', fontsize=13, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add values on bars
        for i, (T, val) in enumerate(zip(self.temperatures, final_J_values)):
            ax4.text(i, val, f'{val:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        
        # Plot 5: Final potentials from each temperature (full width)
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=2, label='V=0 initial')
        for T in self.temperatures:
            V_final = self.V_history[T][-1]
            ax5.plot(self.x, V_final, '-', color=temp_colors[T], 
                    linewidth=2.5, alpha=0.7, label=f'T={T:.2e}')
        
        ax5.set_xlabel('Position x', fontsize=12)
        ax5.set_ylabel('V(x)', fontsize=12)
        ax5.set_title('Final Potentials (All T)', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=9, ncol=6)
        ax5.grid(True, alpha=0.3)

        # Plot 6: Final gradients (full width)
        ax6 = fig.add_subplot(gs[4, :])
        ax6.axhline(y=-1, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Constraint')
        ax6.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=2)
        ax6.axhline(y=0, color='k', linestyle=':', alpha=0.3)

        for T in self.temperatures:
            V_final = self.V_history[T][-1]
            dV_final = self.compute_derivative(V_final)
            ax6.plot(self.x, dV_final, '-', color=temp_colors[T],
                    linewidth=2, alpha=0.7, label=f'T={T:.2e}')

        ax6.set_xlabel('Position x', fontsize=12)
        ax6.set_ylabel('∂V/∂x', fontsize=12)
        ax6.set_title('Final Gradients: -1 < ∂V/∂x < 1', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=9, ncol=6)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(-1.2, 1.2)

        fig.suptitle('Parallel HMMC: All Temperatures Comparison\n' + 
                    'Gradient Constraint: -1 < ∂V/∂x < 1',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current/HMMC_V5_results.png', 
                   dpi=300, bbox_inches='tight')
        print("\nPlot saved: HMMC_V5_results.png")
        return fig

    ''' 
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
                    color=temp_colors[T], label=f'T={T:.2e}', edgecolor='black')
        
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
                        color=temp_colors[T], linewidth=2, alpha=0.7, label=f'T={T:.2e}')
        
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
            
            stats_text += f"T = {T:.2e}:\n"
            stats_text += f"  Final J/P₀:  {final_J:10.4f}\n"
            stats_text += f"  Max J/P₀:    {max_J:10.4f}\n"
            stats_text += f"  Mean (last): {mean_J:10.4f}\n"
            stats_text += f"  Acceptance:  {final_acc:10.3f}\n\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('/Users/Adele/Downloads/Systematic Ratchet Current/HMMC_V5_diagnostics.png',
                   dpi=300, bbox_inches='tight')
        print("Diagnostics saved: parallel_tempering_diagnostics.png")
        return fig
'''

def main():
    """Run parallel tempering MCMC with fixed temperatures"""
    
    print("="*80)
    print("PARALLEL TEMPERING MCMC OPTIMIZATION")
    print("Multiple chains at FIXED temperatures")
    print("="*80)
    
    # Initialize with 5 temperatures spanning low to ~10^6 scale
    # Temperature values on par with max J values for 10-20% acceptance rate
    temperatures = [0.01, 0.1, 0.5, 1.0, 2.0]
    
    # Setting constant
    optimizer = ParallelTemperingOptimizer(
        L=100.0,
        alpha=1.0,
        n_points=1000,
        temperatures=temperatures
    )
    
    # Initial potential
    V_init = np.zeros(optimizer.n_points)

    # Find temperature s.t. acceptance rate is 10 - 20%
    T_opt = optimizer.tune_temperature(V_init=V_init, n_samples=500,
                                       perturbation_scale=0.1)

    # Build temperature ladder around T_opt spanning one decade either side
    if T_opt is not None:
        temperatures = [T_opt * f for f in [0.1, 0.3, 1.0, 3.0, 10.0]]
        print(f"\nUpdated temperature ladder: {[f'{T:.3e}' for T in temperatures]}")
        optimizer.temperatures = temperatures
        optimizer.n_chains = len(temperatures)
        # Reset tracking histories for new temperatures
        optimizer.V_history = {T: [] for T in temperatures}
        optimizer.J_history = {T: [] for T in temperatures}
        optimizer.acceptance_history = {T: [] for T in temperatures}
    
    # Run parallel tempering
    V_best, J_best = optimizer.run_parallel_tempering(
        n_iterations=10000,
        V_init=V_init,
        perturbation_scale=0.1,
        save_every=1
    )
    
    # Plot results
    optimizer.plot_all_temperatures(V_init, V_best)

    # Fourier analysis of the best potential found
    print("\nRunning Fourier analysis on best potential...")
    optimizer.plot_fourier(V_best, n_modes=20)
    
    #print("\nGenerating diagnostics...")
    #optimizer.plot_diagnostics_all_temps()
    
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
