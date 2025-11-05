import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.linear_model import Lasso
from numpy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

# Set plot style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

# ------------------------------
# Load Data (skip first row/column)
# ------------------------------
def load_matrix(filename):
    """Load matrix from Excel file."""
    try:
        df = pd.read_excel(filename, header=0, index_col=0)
        return df.to_numpy()
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found!")
        raise
    except Exception as e:
        print(f"❌ Error loading '{filename}': {e}")
        raise

print("=" * 60)
print("  EEG SOURCE RECONSTRUCTION ANALYSIS")
print("=" * 60)
print("\n📂 Loading Data Files...")

S_true = load_matrix("Source.xlsx")       # (16 × 1000)
G = load_matrix("Leadfield.xlsx")         # (32 × 16)
Y = load_matrix("EEG Data.xlsx")          # (32 × 1000)
Noise = load_matrix("Noise.xlsx")         # (32 × 1000)

print(f"✓ Source Matrix (S_true): {S_true.shape}")
print(f"✓ Leadfield Matrix (G): {G.shape}")
print(f"✓ EEG Data (Y): {Y.shape}")
print(f"✓ Noise Matrix: {Noise.shape}")
print("=" * 60)

# ------------------------------
# Helper functions
# ------------------------------

def weighted_minimum_norm(G, Y, lambda_reg=0.1):
    """
    Weighted Minimum Norm Solution.
    
    Parameters:
    -----------
    G : array (n_sensors × n_sources)
        Leadfield matrix
    Y : array (n_sensors × n_samples)
        EEG measurements
    lambda_reg : float
        Regularization parameter
    
    Returns:
    --------
    S_hat : array (n_sources × n_samples)
        Estimated source activity
    """
    GTG = G.T @ G
    W = np.diag(1 / (np.sqrt(np.diag(GTG)) + 1e-6))  # weight matrix
    inv_term = inv(G.T @ G + (lambda_reg**2) * W @ W)
    S_hat = inv_term @ G.T @ Y
    return S_hat

def sparse_solution(G, Y, alpha=0.001):
    """
    Sparse solution using Lasso (for each time point).
    
    Parameters:
    -----------
    G : array (n_sensors × n_sources)
        Leadfield matrix
    Y : array (n_sensors × n_samples)
        EEG measurements
    alpha : float
        L1 regularization parameter
    
    Returns:
    --------
    S_hat : array (n_sources × n_samples)
        Estimated source activity
    """
    n_sources, n_samples = G.shape[1], Y.shape[1]
    S_hat = np.zeros((n_sources, n_samples))
    G_norm = G / (np.linalg.norm(G, axis=0, keepdims=True) + 1e-12)

    model = Lasso(alpha=alpha, max_iter=5000)
    for i in range(n_samples):
        y_i = Y[:, i]
        if np.allclose(y_i, 0):
            continue
        model.fit(G_norm, y_i)
        S_hat[:, i] = model.coef_
    return S_hat

def accuracy(true, estimated):
    """
    Compute normalized correlation safely (0–1).
    
    Parameters:
    -----------
    true : array
        True source activity
    estimated : array
        Estimated source activity
    
    Returns:
    --------
    corr : float
        Correlation coefficient [0, 1]
    """
    if np.allclose(estimated, 0):
        return 0.0
    corr = np.corrcoef(true.flatten(), estimated.flatten())[0, 1]
    return 0.0 if np.isnan(corr) else corr

def compute_snr(signal, noise):
    """Compute Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

def compute_rmse(true, estimated):
    """Compute Root Mean Square Error."""
    return np.sqrt(np.mean((true - estimated) ** 2))

# ------------------------------
# Experiments
# ------------------------------
conditions = {
    "Without Noise": Y,
    "With Noise": Y + Noise
}

results = {}

# Compute SNR
snr_db = compute_snr(Y, Noise)
print(f"\n📊 Signal-to-Noise Ratio: {snr_db:.2f} dB")

for cond, data in conditions.items():
    print(f"\n{'='*60}")
    print(f"  {cond.upper()}")
    print('='*60)
    
    # Weighted Minimum Norm
    print("⏳ Computing Weighted Minimum Norm...")
    S_hat_wmn = weighted_minimum_norm(G, data)
    acc_wmn = accuracy(S_true, S_hat_wmn)
    rmse_wmn = compute_rmse(S_true, S_hat_wmn)
    print(f"   ✓ Accuracy (Correlation): {acc_wmn:.4f}")
    print(f"   ✓ RMSE: {rmse_wmn:.6f}")

    # Sparse Solution
    print("⏳ Computing Sparse Solution (Lasso)...")
    S_hat_sparse = sparse_solution(G, data)
    acc_sparse = accuracy(S_true, S_hat_sparse)
    rmse_sparse = compute_rmse(S_true, S_hat_sparse)
    print(f"   ✓ Accuracy (Correlation): {acc_sparse:.4f}")
    print(f"   ✓ RMSE: {rmse_sparse:.6f}")

    results[cond] = {
        "WMN": acc_wmn,
        "Sparse": acc_sparse,
        "S_hat_WMN": S_hat_wmn,
        "S_hat_Sparse": S_hat_sparse,
        "RMSE_WMN": rmse_wmn,
        "RMSE_Sparse": rmse_sparse
    }

print("\n" + "="*60)

# ------------------------------
# Interactive Visualization
# ------------------------------
print("\n🎨 Generating Interactive Visualization...")

sample_source = 0     # source index to visualize
window_size = 200     # samples per window
num_samples = S_true.shape[1]
max_start = num_samples - window_size

# Initial window
start_idx = 0
end_idx = start_idx + window_size

fig = plt.figure(figsize=(14, 8))
plt.subplots_adjust(bottom=0.2, top=0.92, hspace=0.35, left=0.08, right=0.95)

# Add main title
fig.suptitle('EEG Source Reconstruction Analysis', fontsize=16, fontweight='bold')

# Create subplots
axes = [plt.subplot(2, 1, i+1) for i in range(2)]

# Color scheme
colors = {
    'true': '#2E86AB',
    'wmn': '#A23B72',
    'sparse': '#F18F01'
}

# Plot initial windows - Without Noise
line_true1, = axes[0].plot(range(window_size), 
                           S_true[sample_source, start_idx:end_idx], 
                           label="True Source", linewidth=2.5, color=colors['true'])
line_wmn1, = axes[0].plot(range(window_size), 
                          results["Without Noise"]["S_hat_WMN"][sample_source, start_idx:end_idx], 
                          '--', label="WMN", linewidth=2, alpha=0.8, color=colors['wmn'])
line_sparse1, = axes[0].plot(range(window_size), 
                             results["Without Noise"]["S_hat_Sparse"][sample_source, start_idx:end_idx], 
                             '-.', label="Sparse (Lasso)", linewidth=2, alpha=0.8, color=colors['sparse'])
axes[0].set_title(f"Reconstruction WITHOUT Noise | Accuracy: WMN={results['Without Noise']['WMN']:.3f}, Sparse={results['Without Noise']['Sparse']:.3f}", 
                  fontsize=11, fontweight='bold')
axes[0].set_ylabel('Amplitude', fontsize=10)
axes[0].legend(loc='upper right', framealpha=0.9)
axes[0].grid(True, alpha=0.3)

# Plot initial windows - With Noise
line_true2, = axes[1].plot(range(window_size), 
                           S_true[sample_source, start_idx:end_idx], 
                           label="True Source", linewidth=2.5, color=colors['true'])
line_wmn2, = axes[1].plot(range(window_size), 
                          results["With Noise"]["S_hat_WMN"][sample_source, start_idx:end_idx], 
                          '--', label="WMN", linewidth=2, alpha=0.8, color=colors['wmn'])
line_sparse2, = axes[1].plot(range(window_size), 
                             results["With Noise"]["S_hat_Sparse"][sample_source, start_idx:end_idx], 
                             '-.', label="Sparse (Lasso)", linewidth=2, alpha=0.8, color=colors['sparse'])
axes[1].set_title(f"Reconstruction WITH Noise (SNR={snr_db:.2f} dB) | Accuracy: WMN={results['With Noise']['WMN']:.3f}, Sparse={results['With Noise']['Sparse']:.3f}", 
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Time Samples', fontsize=10)
axes[1].set_ylabel('Amplitude', fontsize=10)
axes[1].legend(loc='upper right', framealpha=0.9)
axes[1].grid(True, alpha=0.3)

# Slider for scrolling
ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
slider = Slider(ax_slider, 'Time Window Start', 0, max_start, 
                valinit=0, valstep=10, color=colors['wmn'])

# Text box for displaying current position
ax_text = plt.axes([0.82, 0.08, 0.13, 0.03])
ax_text.axis('off')
info_text = ax_text.text(0, 0.5, f'Samples: {start_idx}-{end_idx}', 
                         fontsize=9, va='center')

def update(val):
    """Update plot when slider moves."""
    start = int(slider.val)
    end = start + window_size
    
    # Update lines
    line_true1.set_ydata(S_true[sample_source, start:end])
    line_wmn1.set_ydata(results["Without Noise"]["S_hat_WMN"][sample_source, start:end])
    line_sparse1.set_ydata(results["Without Noise"]["S_hat_Sparse"][sample_source, start:end])
    line_true2.set_ydata(S_true[sample_source, start:end])
    line_wmn2.set_ydata(results["With Noise"]["S_hat_WMN"][sample_source, start:end])
    line_sparse2.set_ydata(results["With Noise"]["S_hat_Sparse"][sample_source, start:end])
    
    # Update info text
    info_text.set_text(f'Samples: {start}-{end}')
    
    # Rescale axes
    for ax in axes:
        ax.relim()
        ax.autoscale_view()
    fig.canvas.draw_idle()

slider.on_changed(update)

# Add reset button
ax_button = plt.axes([0.08, 0.08, 0.08, 0.03])
button = Button(ax_button, 'Reset', color='lightgray', hovercolor='gray')

def reset(event):
    """Reset slider to initial position."""
    slider.reset()

button.on_clicked(reset)

print("✓ Visualization ready! Use the slider to explore different time windows.")
plt.show()

# ------------------------------
# Summary Table
# ------------------------------
print("\n" + "="*60)
print("  ACCURACY SUMMARY")
print("="*60)

summary = pd.DataFrame({
    "Condition": ["Without Noise", "With Noise"],
    "WMN (Correlation)": [results["Without Noise"]["WMN"], results["With Noise"]["WMN"]],
    "Sparse (Correlation)": [results["Without Noise"]["Sparse"], results["With Noise"]["Sparse"]],
    "WMN (RMSE)": [results["Without Noise"]["RMSE_WMN"], results["With Noise"]["RMSE_WMN"]],
    "Sparse (RMSE)": [results["Without Noise"]["RMSE_Sparse"], results["With Noise"]["RMSE_Sparse"]]
})

print(summary.to_string(index=False))
print("="*60)

# Performance comparison
print("\n📈 Performance Analysis:")
print("-" * 60)
for cond in ["Without Noise", "With Noise"]:
    wmn_acc = results[cond]["WMN"]
    sparse_acc = results[cond]["Sparse"]
    
    if wmn_acc > sparse_acc:
        winner = "WMN"
        diff = wmn_acc - sparse_acc
    else:
        winner = "Sparse (Lasso)"
        diff = sparse_acc - wmn_acc
    
    print(f"{cond}:")
    print(f"  🏆 Best Method: {winner} (Δ = {diff:.4f})")

print("-" * 60)

# Noise impact analysis
wmn_degradation = results["Without Noise"]["WMN"] - results["With Noise"]["WMN"]
sparse_degradation = results["Without Noise"]["Sparse"] - results["With Noise"]["Sparse"]

print(f"\n🔊 Noise Impact Analysis:")
print(f"  • WMN accuracy degradation: {wmn_degradation:.4f} ({wmn_degradation/results['Without Noise']['WMN']*100:.2f}%)")
print(f"  • Sparse accuracy degradation: {sparse_degradation:.4f} ({sparse_degradation/results['Without Noise']['Sparse']*100:.2f}%)")

if abs(wmn_degradation) < abs(sparse_degradation):
    print(f"  ✓ WMN is more robust to noise")
else:
    print(f"  ✓ Sparse method is more robust to noise")

print("\n" + "="*60)
print("✅ Analysis Complete!")
print("="*60)
