import matplotlib.pyplot as plt
import numpy as np

# 1. IEEE Standard Formatting Configuration (Scaled down text sizes)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 8,       # Reduced from 10
    'font.size': 8,            # Reduced from 10
    'legend.fontsize': 7,      # Reduced from 9
    'xtick.labelsize': 7,      # Reduced from 9
    'ytick.labelsize': 7,      # Reduced from 9
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7
})

# 2. Extract Data (from mass-flow-inlet-rfile.out)
iterations = np.arange(1, 138)
mass_flow = np.array([
    0.010772, 0.006575, 0.003072, 0.001750, 0.000468, -0.000401, 0.000074, -0.000361, -0.000279, 0.000397,
    0.000520, 0.000924, 0.001414, 0.001948, 0.002685, 0.003167, 0.003948, 0.004289, 0.004444, 0.004772,
    0.005176, 0.005169, 0.004978, 0.005066, 0.005181, 0.005406, 0.005626, 0.005836, 0.005932, 0.006033,
    0.006177, 0.006290, 0.006400, 0.006452, 0.006502, 0.006572, 0.006634, 0.006704, 0.006752, 0.006804,
    0.006863, 0.006909, 0.006957, 0.007013, 0.007062, 0.007114, 0.007182, 0.007243, 0.007309, 0.007367,
    0.007460, 0.007540, 0.007659, 0.007712, 0.007823, 0.007907, 0.008008, 0.008083, 0.008147, 0.008274,
    0.008340, 0.008405, 0.008508, 0.008582, 0.008690, 0.008742, 0.008810, 0.008888, 0.008956, 0.009045,
    0.009085, 0.009140, 0.009195, 0.009262, 0.009314, 0.009356, 0.009397, 0.009444, 0.009472, 0.009510,
    0.009538, 0.009572, 0.009592, 0.009618, 0.009641, 0.009662, 0.009683, 0.009700, 0.009717, 0.009732,
    0.009748, 0.009760, 0.009772, 0.009783, 0.009793, 0.009805, 0.009814, 0.009823, 0.009833, 0.009842,
    0.009850, 0.009854, 0.009862, 0.009869, 0.009875, 0.009881, 0.009887, 0.009893, 0.009899, 0.009905,
    0.009911, 0.009917, 0.009924, 0.009929, 0.009935, 0.009941, 0.009949, 0.009952, 0.009959, 0.009965,
    0.009971, 0.009979, 0.009984, 0.009990, 0.009997, 0.010003, 0.010010, 0.010017, 0.010024, 0.010030,
    0.010037, 0.010044, 0.010050, 0.010056, 0.010062, 0.010070, 0.010077
])

target_mass_flow = 0.0101

# 3. Create Figure (Single Column IEEE Width: 3.5 inches x 2.5 inches)
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# 4. Plot Data
ax.plot(iterations, mass_flow, color='k', linewidth=1.5, label='CFD Inlet Mass Flow')
ax.axhline(y=target_mass_flow, color='r', linestyle='--', linewidth=1.2, label='Target (0.0101 kg/s)')

# 5. Formatting & Labels
ax.set_xlabel('Iteration')
ax.set_ylabel('Mass Flow Rate (kg/s)')
ax.set_title('Inlet Mass Flow Rate Convergence', fontsize=9, pad=8)  # Reduced title font

# 6. Legend Configuration 
ax.legend(loc='lower right', framealpha=0.9, edgecolor='0.8')

# Apply tight layout to prevent clipping
plt.tight_layout()

# Save and Show
plt.savefig('mass_flow_trend_ieee_smaller_text.png', dpi=300)
plt.show()