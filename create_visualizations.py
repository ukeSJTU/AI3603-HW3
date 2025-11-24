# -*- coding:utf-8 -*-

"""
Create additional visualizations for better report presentation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add legend
from matplotlib.patches import Circle, Patch

from BayesianNetworks import inference, readFactorTablefromData

# Set style
sns.set_style("whitegrid")

# Load data
riskFactorNet = pd.read_csv('data/RiskFactorsData.csv')

print("Creating additional visualizations...")

###########################################################################
# Visualization 1: Network structure comparison
###########################################################################
print("\n1. Creating network structure diagrams...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Bayesian Network Structures Comparison', fontsize=16, fontweight='bold')

# Network 1 (original)
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('Network 1: Original\n(No direct habit→outcome links)', fontsize=12, fontweight='bold')

# Draw nodes and edges for network 1
nodes = {
    'income': (5, 11),
    'smoke': (2, 9),
    'exercise': (4, 9),
    'long_sit': (6, 9),
    'stay_up': (8, 9),
    'bmi': (5, 7),
    'cholesterol': (3, 5),
    'bp': (7, 5),
    'diabetes': (2, 3),
    'stroke': (4, 3),
    'attack': (6, 3),
    'angina': (8, 3)
}

# Draw edges
edges = [
    ('income', 'smoke'), ('income', 'exercise'), ('income', 'long_sit'), 
    ('income', 'stay_up'), ('income', 'bmi'),
    ('smoke', 'cholesterol'), ('exercise', 'cholesterol'),
    ('bmi', 'bp'), ('bmi', 'diabetes'),
    ('bp', 'stroke'), ('bp', 'attack'), ('bp', 'angina'),
    ('cholesterol', 'stroke'), ('cholesterol', 'attack'), ('cholesterol', 'angina')
]

for start, end in edges:
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

# Draw nodes
for node, (x, y) in nodes.items():
    if node == 'income':
        color = '#ff9999'
    elif node in ['smoke', 'exercise', 'long_sit', 'stay_up']:
        color = '#ffcc99'
    elif node in ['bmi', 'bp', 'cholesterol']:
        color = '#99ccff'
    else:
        color = '#99ff99'
    ax.add_patch(Circle((x, y), 0.4, color=color, ec='black', lw=2))
    ax.text(x, y, node.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')

# Network 2 (with habit links)
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('Network 2: With Habit Links\n(smoke, exercise → outcomes)', fontsize=12, fontweight='bold')

# Draw edges (including new ones)
edges2 = edges + [
    ('smoke', 'diabetes'), ('smoke', 'stroke'), ('smoke', 'attack'), ('smoke', 'angina'),
    ('exercise', 'diabetes'), ('exercise', 'stroke'), ('exercise', 'attack'), ('exercise', 'angina')
]

for start, end in edges2:
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    # Highlight new edges
    if (start, end) not in edges:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    else:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

# Draw nodes
for node, (x, y) in nodes.items():
    if node == 'income':
        color = '#ff9999'
    elif node in ['smoke', 'exercise', 'long_sit', 'stay_up']:
        color = '#ffcc99'
    elif node in ['bmi', 'bp', 'cholesterol']:
        color = '#99ccff'
    else:
        color = '#99ff99'
    ax.add_patch(Circle((x, y), 0.4, color=color, ec='black', lw=2))
    ax.text(x, y, node.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')

# Network 3 (with outcome interaction)
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('Network 3: With Outcome Link\n(diabetes → stroke)', fontsize=12, fontweight='bold')

# Draw edges (including diabetes->stroke)
edges3 = edges2 + [('diabetes', 'stroke')]

for start, end in edges3:
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    # Highlight new edge
    if start == 'diabetes' and end == 'stroke':
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='blue'))
    elif (start, end) not in edges:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    else:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

# Draw nodes
for node, (x, y) in nodes.items():
    if node == 'income':
        color = '#ff9999'
    elif node in ['smoke', 'exercise', 'long_sit', 'stay_up']:
        color = '#ffcc99'
    elif node in ['bmi', 'bp', 'cholesterol']:
        color = '#99ccff'
    else:
        color = '#99ff99'
    ax.add_patch(Circle((x, y), 0.4, color=color, ec='black', lw=2))
    ax.text(x, y, node.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')

legend_elements = [
    Patch(facecolor='#ff9999', edgecolor='black', label='Root (Income)'),
    Patch(facecolor='#ffcc99', edgecolor='black', label='Habits'),
    Patch(facecolor='#99ccff', edgecolor='black', label='Health Indicators'),
    Patch(facecolor='#99ff99', edgecolor='black', label='Health Outcomes'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # type: ignore
plt.savefig('assets/network_structures.png', dpi=300, bbox_inches='tight')
print("   Saved: assets/network_structures.png")

###########################################################################
# Visualization 2: Probability comparison heatmap
###########################################################################
print("\n2. Creating probability comparison heatmap...")

# Create all networks
income = readFactorTablefromData(riskFactorNet, ['income'])
smoke = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
bp = readFactorTablefromData(riskFactorNet, ['bp', 'bmi'])
cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol', 'smoke', 'exercise'])
diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
stroke = readFactorTablefromData(riskFactorNet, ['stroke', 'bp', 'cholesterol'])
attack = readFactorTablefromData(riskFactorNet, ['attack', 'bp', 'cholesterol'])
angina = readFactorTablefromData(riskFactorNet, ['angina', 'bp', 'cholesterol'])

risk_net1 = [income, smoke, exercise, long_sit, stay_up, bmi, bp, cholesterol, 
             diabetes, stroke, attack, angina]

# Network 2
cholesterol2 = readFactorTablefromData(riskFactorNet, ['cholesterol', 'smoke', 'exercise'])
diabetes2 = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'smoke', 'exercise'])
stroke2 = readFactorTablefromData(riskFactorNet, ['stroke', 'bp', 'cholesterol', 'smoke', 'exercise'])
attack2 = readFactorTablefromData(riskFactorNet, ['attack', 'bp', 'cholesterol', 'smoke', 'exercise'])
angina2 = readFactorTablefromData(riskFactorNet, ['angina', 'bp', 'cholesterol', 'smoke', 'exercise'])

risk_net2 = [income, smoke, exercise, long_sit, stay_up, bmi, bp, cholesterol2, 
             diabetes2, stroke2, attack2, angina2]

# Create comparison matrix
outcomes = ['diabetes', 'stroke', 'attack', 'angina']
conditions = ['Baseline', 'Bad Habits', 'Good Habits', 'Poor Health', 'Good Health']
data_net1 = []
data_net2 = []

all_vars = ['income', 'smoke', 'exercise', 'long_sit', 'stay_up', 'bmi', 
            'bp', 'cholesterol', 'diabetes', 'stroke', 'attack', 'angina']

for outcome in outcomes:
    margVars = [v for v in all_vars if v != outcome]
    row_net1 = []
    row_net2 = []
    
    # Baseline (no evidence)
    p1 = inference(risk_net1, margVars, [], [])
    p2 = inference(risk_net2, margVars, [], [])
    row_net1.append(p1[p1[outcome] == 1]['probs'].values[0] if 1 in p1[outcome].values else 0)
    row_net2.append(p2[p2[outcome] == 1]['probs'].values[0] if 1 in p2[outcome].values else 0)
    
    # Bad habits
    p1 = inference(risk_net1, margVars, ['smoke', 'exercise', 'long_sit', 'stay_up'], [1, 2, 1, 1])
    p2 = inference(risk_net2, margVars, ['smoke', 'exercise', 'long_sit', 'stay_up'], [1, 2, 1, 1])
    row_net1.append(p1[p1[outcome] == 1]['probs'].values[0] if 1 in p1[outcome].values else 0)
    row_net2.append(p2[p2[outcome] == 1]['probs'].values[0] if 1 in p2[outcome].values else 0)
    
    # Good habits
    p1 = inference(risk_net1, margVars, ['smoke', 'exercise', 'long_sit', 'stay_up'], [2, 1, 2, 2])
    p2 = inference(risk_net2, margVars, ['smoke', 'exercise', 'long_sit', 'stay_up'], [2, 1, 2, 2])
    row_net1.append(p1[p1[outcome] == 1]['probs'].values[0] if 1 in p1[outcome].values else 0)
    row_net2.append(p2[p2[outcome] == 1]['probs'].values[0] if 1 in p2[outcome].values else 0)
    
    # Poor health
    p1 = inference(risk_net1, margVars, ['bp', 'cholesterol', 'bmi'], [1, 1, 4])
    p2 = inference(risk_net2, margVars, ['bp', 'cholesterol', 'bmi'], [1, 1, 4])
    row_net1.append(p1[p1[outcome] == 1]['probs'].values[0] if 1 in p1[outcome].values else 0)
    row_net2.append(p2[p2[outcome] == 1]['probs'].values[0] if 1 in p2[outcome].values else 0)
    
    # Good health
    p1 = inference(risk_net1, margVars, ['bp', 'cholesterol', 'bmi'], [3, 2, 2])
    p2 = inference(risk_net2, margVars, ['bp', 'cholesterol', 'bmi'], [3, 2, 2])
    row_net1.append(p1[p1[outcome] == 1]['probs'].values[0] if 1 in p1[outcome].values else 0)
    row_net2.append(p2[p2[outcome] == 1]['probs'].values[0] if 1 in p2[outcome].values else 0)
    
    data_net1.append(row_net1)
    data_net2.append(row_net2)

# Create heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Probability Heatmaps: Network Comparison', fontsize=16, fontweight='bold')

# Network 1
ax = axes[0]
df1 = pd.DataFrame(data_net1, index=[o.capitalize() for o in outcomes], columns=conditions)
sns.heatmap(df1, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Probability'})
ax.set_title('Network 1 (Original)', fontsize=12, fontweight='bold')
ax.set_xlabel('Condition', fontsize=10)
ax.set_ylabel('Outcome', fontsize=10)

# Network 2
ax = axes[1]
df2 = pd.DataFrame(data_net2, index=[o.capitalize() for o in outcomes], columns=conditions)
sns.heatmap(df2, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Probability'})
ax.set_title('Network 2 (With Habit Links)', fontsize=12, fontweight='bold')
ax.set_xlabel('Condition', fontsize=10)
ax.set_ylabel('Outcome', fontsize=10)

plt.tight_layout()
plt.savefig('assets/probability_heatmaps.png', dpi=300, bbox_inches='tight')
print("   Saved: assets/probability_heatmaps.png")

###########################################################################
# Visualization 3: Difference analysis
###########################################################################
print("\n3. Creating difference analysis...")

fig, ax = plt.subplots(figsize=(12, 8))

# Calculate differences
differences = np.array(data_net2) - np.array(data_net1)
df_diff = pd.DataFrame(differences, index=[o.capitalize() for o in outcomes], columns=conditions)

# Create heatmap
sns.heatmap(df_diff, annot=True, fmt='+.4f', cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'label': 'Probability Difference (Network 2 - Network 1)'})
ax.set_title('Impact of Adding Direct Habit→Outcome Links', fontsize=14, fontweight='bold')
ax.set_xlabel('Condition', fontsize=12)
ax.set_ylabel('Outcome', fontsize=12)

plt.tight_layout()
plt.savefig('assets/network_difference.png', dpi=300, bbox_inches='tight')
print("   Saved: assets/network_difference.png")

print("\n" + "=" * 60)
print("All additional visualizations created successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  1. assets/network_structures.png - Network architecture comparison")
print("  2. assets/probability_heatmaps.png - Probability comparison")
print("  3. assets/network_difference.png - Impact of model changes")
