# -*- coding:utf-8 -*-

"""
Bayesian Network Analysis for Risk Factors
This script answers all questions from the homework assignment
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from BayesianNetworks import (
    inference,
    readFactorTablefromData,
)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
print("Loading data...")
riskFactorNet = pd.read_csv('data/RiskFactorsData.csv')
print(f"Data loaded: {len(riskFactorNet)} records\n")

###########################################################################
# Question 1: Network Structure and Size
###########################################################################
print("=" * 80)
print("QUESTION 1: Network Structure and Size")
print("=" * 80)

# Create the Bayesian network factors
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

# Count probabilities in the network
network_size = 0
network_size += len(income)  # P(income): 8 values
network_size += len(smoke)   # P(smoke|income): 2*8 = 16 values
network_size += len(exercise)  # P(exercise|income): 2*8 = 16 values
network_size += len(long_sit)  # P(long_sit|income): 2*8 = 16 values
network_size += len(stay_up)   # P(stay_up|income): 2*8 = 16 values
network_size += len(bmi)       # P(bmi|income): 4*8 = 32 values
network_size += len(bp)        # P(bp|bmi): 4*4 = 16 values
network_size += len(cholesterol)  # P(cholesterol|smoke,exercise): 2*2*2 = 8 values
network_size += len(diabetes)  # P(diabetes|bmi): 4*4 = 16 values
network_size += len(stroke)    # P(stroke|bp,cholesterol): 2*4*2 = 16 values
network_size += len(attack)    # P(attack|bp,cholesterol): 2*4*2 = 16 values
network_size += len(angina)    # P(angina|bp,cholesterol): 2*4*2 = 16 values

print(f"Network size (number of probabilities): {network_size}")

# Calculate full joint distribution size
# Variables and their values:
# income: 8, smoke: 2, exercise: 2, long_sit: 2, stay_up: 2, bmi: 4
# bp: 4, cholesterol: 2, diabetes: 4, stroke: 2, attack: 2, angina: 2
full_joint_size = 8 * 2 * 2 * 2 * 2 * 4 * 4 * 2 * 4 * 2 * 2 * 2
print(f"Full joint distribution size: {full_joint_size}")
print(f"Compression ratio: {full_joint_size / network_size:.2f}x\n")

###########################################################################
# Question 2: Health Outcomes Analysis
###########################################################################
print("=" * 80)
print("QUESTION 2: Health Outcomes Analysis")
print("=" * 80)

# Create the Bayesian network
risk_net = [income, smoke, exercise, long_sit, stay_up, bmi, bp, cholesterol, 
            diabetes, stroke, attack, angina]

outcomes = ['diabetes', 'stroke', 'attack', 'angina']

print("\n2(a): Probability of outcomes with bad vs good habits")
print("-" * 60)

# Bad habits: smoke=1, exercise=2, long_sit=1, stay_up=1
# Good habits: smoke=2, exercise=1, long_sit=2, stay_up=2

results_2a = []

for outcome in outcomes:
    # Get all variables except the outcome
    all_vars = ['income', 'smoke', 'exercise', 'long_sit', 'stay_up', 'bmi', 
                'bp', 'cholesterol', 'diabetes', 'stroke', 'attack', 'angina']
    margVars = [v for v in all_vars if v != outcome]
    
    # Bad habits
    obsVars_bad = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals_bad = [1, 2, 1, 1]
    p_bad = inference(risk_net, margVars, obsVars_bad, obsVals_bad)
    prob_bad = p_bad[p_bad[outcome] == 1]['probs'].values[0] if 1 in p_bad[outcome].values else 0
    
    # Good habits
    obsVars_good = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals_good = [2, 1, 2, 2]
    p_good = inference(risk_net, margVars, obsVars_good, obsVals_good)
    prob_good = p_good[p_good[outcome] == 1]['probs'].values[0] if 1 in p_good[outcome].values else 0
    
    print(f"{outcome.capitalize():10s} | Bad habits: {prob_bad:.6f} | Good habits: {prob_good:.6f} | Difference: {prob_bad - prob_good:+.6f}")
    results_2a.append([outcome, prob_bad, prob_good])

print("\n2(b): Probability of outcomes with poor vs good health")
print("-" * 60)

# Poor health: bp=1 (high), cholesterol=1 (high), bmi=4 (obese)
# Good health: bp=3 (no high BP), cholesterol=2 (no high cholesterol), bmi=2 (normal)

results_2b = []

for outcome in outcomes:
    all_vars = ['income', 'smoke', 'exercise', 'long_sit', 'stay_up', 'bmi', 
                'bp', 'cholesterol', 'diabetes', 'stroke', 'attack', 'angina']
    margVars = [v for v in all_vars if v != outcome]
    
    # Poor health
    obsVars_poor = ['bp', 'cholesterol', 'bmi']
    obsVals_poor = [1, 1, 4]
    p_poor = inference(risk_net, margVars, obsVars_poor, obsVals_poor)
    prob_poor = p_poor[p_poor[outcome] == 1]['probs'].values[0] if 1 in p_poor[outcome].values else 0
    
    # Good health
    obsVars_good = ['bp', 'cholesterol', 'bmi']
    obsVals_good = [3, 2, 2]
    p_good = inference(risk_net, margVars, obsVars_good, obsVals_good)
    prob_good = p_good[p_good[outcome] == 1]['probs'].values[0] if 1 in p_good[outcome].values else 0
    
    print(f"{outcome.capitalize():10s} | Poor health: {prob_poor:.6f} | Good health: {prob_good:.6f} | Difference: {prob_poor - prob_good:+.6f}")
    results_2b.append([outcome, prob_poor, prob_good])

###########################################################################
# Question 3: Income Effect Analysis
###########################################################################
print("\n" + "=" * 80)
print("QUESTION 3: Income Effect Analysis")
print("=" * 80)

# For each outcome, compute P(outcome=1 | income=i) for i=1..8
income_effects = {outcome: [] for outcome in outcomes}

for outcome in outcomes:
    all_vars = ['income', 'smoke', 'exercise', 'long_sit', 'stay_up', 'bmi', 
                'bp', 'cholesterol', 'diabetes', 'stroke', 'attack', 'angina']
    margVars = [v for v in all_vars if v != outcome]
    
    for income_val in range(1, 9):
        p = inference(risk_net, margVars, ['income'], [income_val])
        prob = p[p[outcome] == 1]['probs'].values[0] if 1 in p[outcome].values else 0
        income_effects[outcome].append(prob)
        print(f"P({outcome}=1 | income={income_val}): {prob:.6f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Effect of Income on Health Outcomes', fontsize=16, fontweight='bold')

income_labels = ['<10K', '10-15K', '15-20K', '20-25K', '25-35K', '35-50K', '50-75K', '>75K']

for idx, outcome in enumerate(outcomes):
    ax = axes[idx // 2, idx % 2]
    ax.plot(range(1, 9), income_effects[outcome], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Income Level', fontsize=12)
    ax.set_ylabel(f'P({outcome}=1 | income)', fontsize=12)
    ax.set_title(f'{outcome.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(income_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/income_effect.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to assets/income_effect.png")

###########################################################################
# Question 4: Testing Independence Assumptions
###########################################################################
print("\n" + "=" * 80)
print("QUESTION 4: Testing Independence Assumptions")
print("=" * 80)

# Create second network with edges from smoking and exercise to outcomes
print("\nCreating second network with direct edges from habits to outcomes...")

# For the second network, we add direct dependencies
cholesterol2 = readFactorTablefromData(riskFactorNet, ['cholesterol', 'smoke', 'exercise'])
diabetes2 = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'smoke', 'exercise'])
stroke2 = readFactorTablefromData(riskFactorNet, ['stroke', 'bp', 'cholesterol', 'smoke', 'exercise'])
attack2 = readFactorTablefromData(riskFactorNet, ['attack', 'bp', 'cholesterol', 'smoke', 'exercise'])
angina2 = readFactorTablefromData(riskFactorNet, ['angina', 'bp', 'cholesterol', 'smoke', 'exercise'])

risk_net2 = [income, smoke, exercise, long_sit, stay_up, bmi, bp, cholesterol2, 
             diabetes2, stroke2, attack2, angina2]

print("\nRe-doing queries from Question 2(a) with new network:")
print("-" * 60)

for outcome in outcomes:
    all_vars = ['income', 'smoke', 'exercise', 'long_sit', 'stay_up', 'bmi', 
                'bp', 'cholesterol', 'diabetes', 'stroke', 'attack', 'angina']
    margVars = [v for v in all_vars if v != outcome]
    
    # Bad habits
    obsVars_bad = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals_bad = [1, 2, 1, 1]
    p_bad = inference(risk_net2, margVars, obsVars_bad, obsVals_bad)
    prob_bad_new = p_bad[p_bad[outcome] == 1]['probs'].values[0] if 1 in p_bad[outcome].values else 0
    
    # Good habits
    obsVars_good = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals_good = [2, 1, 2, 2]
    p_good = inference(risk_net2, margVars, obsVars_good, obsVals_good)
    prob_good_new = p_good[p_good[outcome] == 1]['probs'].values[0] if 1 in p_good[outcome].values else 0
    
    # Get original values
    prob_bad_old = [r[1] for r in results_2a if r[0] == outcome][0]
    prob_good_old = [r[2] for r in results_2a if r[0] == outcome][0]
    
    print(f"{outcome.capitalize():10s}")
    print(f"  Old network - Bad: {prob_bad_old:.6f}, Good: {prob_good_old:.6f}, Diff: {prob_bad_old - prob_good_old:+.6f}")
    print(f"  New network - Bad: {prob_bad_new:.6f}, Good: {prob_good_new:.6f}, Diff: {prob_bad_new - prob_good_new:+.6f}")
    print(f"  Change in bad habits impact: {abs((prob_bad_new - prob_good_new) - (prob_bad_old - prob_good_old)):.6f}")

print("\nRe-doing queries from Question 2(b) with new network:")
print("-" * 60)

for outcome in outcomes:
    all_vars = ['income', 'smoke', 'exercise', 'long_sit', 'stay_up', 'bmi', 
                'bp', 'cholesterol', 'diabetes', 'stroke', 'attack', 'angina']
    margVars = [v for v in all_vars if v != outcome]
    
    # Poor health
    obsVars_poor = ['bp', 'cholesterol', 'bmi']
    obsVals_poor = [1, 1, 4]
    p_poor = inference(risk_net2, margVars, obsVars_poor, obsVals_poor)
    prob_poor_new = p_poor[p_poor[outcome] == 1]['probs'].values[0] if 1 in p_poor[outcome].values else 0
    
    # Good health
    obsVars_good = ['bp', 'cholesterol', 'bmi']
    obsVals_good = [3, 2, 2]
    p_good = inference(risk_net2, margVars, obsVars_good, obsVals_good)
    prob_good_new = p_good[p_good[outcome] == 1]['probs'].values[0] if 1 in p_good[outcome].values else 0
    
    # Get original values
    prob_poor_old = [r[1] for r in results_2b if r[0] == outcome][0]
    prob_good_old = [r[2] for r in results_2b if r[0] == outcome][0]
    
    print(f"{outcome.capitalize():10s}")
    print(f"  Old network - Poor: {prob_poor_old:.6f}, Good: {prob_good_old:.6f}, Diff: {prob_poor_old - prob_good_old:+.6f}")
    print(f"  New network - Poor: {prob_poor_new:.6f}, Good: {prob_good_new:.6f}, Diff: {prob_poor_new - prob_good_new:+.6f}")

###########################################################################
# Question 5: Outcome Interactions
###########################################################################
print("\n" + "=" * 80)
print("QUESTION 5: Outcome Interactions")
print("=" * 80)

# Create third network with edge from diabetes to stroke
print("\nCreating third network with edge from diabetes to stroke...")

stroke3 = readFactorTablefromData(riskFactorNet, ['stroke', 'bp', 'cholesterol', 'diabetes', 'smoke', 'exercise'])

risk_net3 = [income, smoke, exercise, long_sit, stay_up, bmi, bp, cholesterol2, 
             diabetes2, stroke3, attack2, angina2]

# Evaluate P(stroke=1 | diabetes=1) and P(stroke=1 | diabetes=3) for both networks
print("\nNetwork 2 (without diabetes->stroke edge):")
all_vars = ['income', 'smoke', 'exercise', 'long_sit', 'stay_up', 'bmi', 
            'bp', 'cholesterol', 'diabetes', 'stroke', 'attack', 'angina']
margVars = [v for v in all_vars if v != 'stroke']

p_stroke_diab1_net2 = inference(risk_net2, margVars, ['diabetes'], [1])
prob_stroke_diab1_net2 = p_stroke_diab1_net2[p_stroke_diab1_net2['stroke'] == 1]['probs'].values[0]

p_stroke_diab3_net2 = inference(risk_net2, margVars, ['diabetes'], [3])
prob_stroke_diab3_net2 = p_stroke_diab3_net2[p_stroke_diab3_net2['stroke'] == 1]['probs'].values[0]

print(f"  P(stroke=1 | diabetes=1) = {prob_stroke_diab1_net2:.6f}")
print(f"  P(stroke=1 | diabetes=3) = {prob_stroke_diab3_net2:.6f}")
print(f"  Difference: {prob_stroke_diab1_net2 - prob_stroke_diab3_net2:+.6f}")

print("\nNetwork 3 (with diabetes->stroke edge):")
p_stroke_diab1_net3 = inference(risk_net3, margVars, ['diabetes'], [1])
prob_stroke_diab1_net3 = p_stroke_diab1_net3[p_stroke_diab1_net3['stroke'] == 1]['probs'].values[0]

p_stroke_diab3_net3 = inference(risk_net3, margVars, ['diabetes'], [3])
prob_stroke_diab3_net3 = p_stroke_diab3_net3[p_stroke_diab3_net3['stroke'] == 1]['probs'].values[0]

print(f"  P(stroke=1 | diabetes=1) = {prob_stroke_diab1_net3:.6f}")
print(f"  P(stroke=1 | diabetes=3) = {prob_stroke_diab3_net3:.6f}")
print(f"  Difference: {prob_stroke_diab1_net3 - prob_stroke_diab3_net3:+.6f}")

print(f"\nChange in effect: {abs((prob_stroke_diab1_net3 - prob_stroke_diab3_net3) - (prob_stroke_diab1_net2 - prob_stroke_diab3_net2)):.6f}")

###########################################################################
# Additional Visualizations
###########################################################################
print("\n" + "=" * 80)
print("Creating additional visualizations...")
print("=" * 80)

# Visualization for Question 2
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Health Outcomes: Bad vs Good Habits/Health', fontsize=16, fontweight='bold')

# Question 2a visualization
ax = axes[0]
outcomes_list = [r[0] for r in results_2a]
bad_habits = [r[1] for r in results_2a]
good_habits = [r[2] for r in results_2a]

x = np.arange(len(outcomes_list))
width = 0.35

ax.bar(x - width/2, bad_habits, width, label='Bad Habits', color='#d62728', alpha=0.8)
ax.bar(x + width/2, good_habits, width, label='Good Habits', color='#2ca02c', alpha=0.8)
ax.set_xlabel('Health Outcome', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Effect of Habits on Health Outcomes', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([o.capitalize() for o in outcomes_list])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Question 2b visualization
ax = axes[1]
outcomes_list = [r[0] for r in results_2b]
poor_health = [r[1] for r in results_2b]
good_health = [r[2] for r in results_2b]

ax.bar(x - width/2, poor_health, width, label='Poor Health', color='#ff7f0e', alpha=0.8)
ax.bar(x + width/2, good_health, width, label='Good Health', color='#1f77b4', alpha=0.8)
ax.set_xlabel('Health Outcome', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Effect of Health Status on Outcomes', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([o.capitalize() for o in outcomes_list])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('assets/habits_health_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved to assets/habits_health_comparison.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nAll visualizations have been saved to the assets/ folder:")
print("  - assets/income_effect.png")
print("  - assets/habits_health_comparison.png")
print("\nYou can use these results and charts for your assignment report.")
