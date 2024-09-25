import matplotlib.pyplot as plt
import numpy as np

before_RW=[[-0.171, 0.356, -0.067]]
after_RW=[[-0.171, 0.31, -0.058]]

# Example data: statistical parity difference and disparate impact values before and after reweighing
plt.rcParams.update({'font.size': 14})

groups = ['Race', 'Sex', 'Ethnicity']
metrics = ['Statistical Parity Difference', 'Disparate Impact', 'Theil Index']
# Set width of bars
bar_width = 0.35

# Set position of bars on X axis
r1 = np.arange(len(groups))
r2 = [x + bar_width for x in r1]

# Plotting first metric: Statistical Parity Difference
fig, ax = plt.subplots(figsize=(10, 6))

# Before reweighing bars
bars1 = ax.bar(r1, before_RW[0, :], color='b', width=bar_width, edgecolor='grey', label='Before Reweighing')
bars2 = ax.bar(r2, after_RW[0, :], color='r', width=bar_width, edgecolor='grey', hatch='//', label='After Reweighing')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Groups')
ax.set_ylabel('Statistical Parity Difference')
ax.set_title('Statistical Parity Difference Before and After Reweighing')
ax.set_xticks([r + bar_width / 2 for r in range(len(groups))])
ax.set_xticklabels(groups)
ax.legend()

# Adjust layout
plt.tight_layout()

# Show plot for Statistical Parity Difference
plt.show()

# Plotting second metric: Disparate Impact
fig, ax = plt.subplots(figsize=(10, 6))

# Before reweighing bars
bars1 = ax.bar(r1, before_RW[1, :], color='b', width=bar_width, edgecolor='grey', label='Before Reweighing')
bars2 = ax.bar(r2, after_RW[1, :], color='r', width=bar_width, edgecolor='grey', hatch='//', label='After Reweighing')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Groups')
ax.set_ylabel('Disparate Impact')
ax.set_title('Disparate Impact Before and After Reweighing')
ax.set_xticks([r + bar_width / 2 for r in range(len(groups))])
ax.set_xticklabels(groups)
ax.legend()

# Adjust layout
plt.tight_layout()

# Show plot for Disparate Impact
plt.show()


# Plotting third metric: Theil Index
fig, ax = plt.subplots(figsize=(10, 6))

# Before reweighing bars
bars1 = ax.bar(r1, before_RW[2, :], color='b', width=bar_width, edgecolor='grey', label='Before Reweighing')
bars2 = ax.bar(r2, after_RW[2, :], color='r', width=bar_width, edgecolor='grey', hatch='//', label='After Reweighing')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Groups')
ax.set_ylabel('Theil Index')
ax.set_title('Theil Index Before and After Reweighing')
ax.set_xticks([r + bar_width / 2 for r in range(len(groups))])
ax.set_xticklabels(groups)
ax.legend()

# Adjust layout
plt.tight_layout()

# Show plot for Theil index
plt.show()


print(before_RW)
print(after_RW)