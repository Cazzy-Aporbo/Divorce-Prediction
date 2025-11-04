import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#0F1618', '#000000', '#4A4682', '#3A5C60', '#4A696E', '#8FC7B8']
GRADIENT = ['#0F1618', '#1F2628', '#2F3638', '#3A5C60', '#4A696E', '#5A7A7E', '#6A8A8E', '#7AAA9E', '#8FC7B8']
OUTPUT_PATH = '/Users/cazandraaporbo/Desktop/mygit/Divorce/'

plt.style.use('dark_background')

def load_prepare():
    df = pd.read_csv('/Users/cazandraaporbo/Desktop/mygit/Divorce/divorce_df.csv')
    
    le = LabelEncoder()
    df_encoded = df.copy()
    for col in ['education_level', 'employment_status', 'religious_compatibility']:
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
    
    return df, df_encoded

def feature_network_graph(df):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0]
    
    numeric_features = ['age_at_marriage', 'marriage_duration_years', 'num_children',
                       'combined_income', 'communication_score', 'conflict_frequency']
    
    corr_matrix = df[numeric_features + ['divorced']].corr()
    
    G = nx.Graph()
    
    for feature in numeric_features:
        G.add_node(feature, node_type='feature')
    G.add_node('divorced', node_type='outcome')
    
    threshold = 0.15
    for i, feat1 in enumerate(numeric_features):
        for feat2 in numeric_features[i+1:]:
            corr = abs(corr_matrix.loc[feat1, feat2])
            if corr > threshold:
                G.add_edge(feat1, feat2, weight=corr)
    
    for feat in numeric_features:
        corr = abs(corr_matrix.loc[feat, 'divorced'])
        if corr > 0.05:
            G.add_edge(feat, 'divorced', weight=corr)
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    node_colors = []
    for node in G.nodes():
        if node == 'divorced':
            node_colors.append(COLORS[2])
        else:
            node_colors.append(COLORS[4])
    
    node_sizes = []
    for node in G.nodes():
        if node == 'divorced':
            node_sizes.append(3000)
        else:
            importance = abs(corr_matrix.loc[node, 'divorced'])
            node_sizes.append(1000 + importance * 2000)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    edge_widths = [w / max_weight * 5 for w in weights]
    edge_colors = [GRADIENT[int(w / max_weight * (len(GRADIENT)-1))] for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color=edge_colors, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.9, edgecolors=COLORS[5], linewidths=2, ax=ax)
    
    labels = {node: node.replace('_', '\n').title() for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_color=COLORS[5],
                           font_weight='bold', ax=ax)
    
    ax.set_facecolor('#0D1214')
    ax.set_title('Feature Relationship Network', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.axis('off')
    
    ax = axes[1]
    
    feature_divorce_corr = corr_matrix['divorced'].drop('divorced').abs().sort_values(ascending=True)
    
    y_pos = np.arange(len(feature_divorce_corr))
    colors_bar = [GRADIENT[int(val / feature_divorce_corr.max() * (len(GRADIENT)-1))] 
                  for val in feature_divorce_corr]
    
    bars = ax.barh(y_pos, feature_divorce_corr.values, color=colors_bar,
                   edgecolor=COLORS[5], linewidth=2, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in feature_divorce_corr.index],
                       fontsize=11, color=COLORS[5])
    ax.set_xlabel('Correlation Strength with Divorce', fontsize=12, color=COLORS[5])
    ax.set_title('Direct Influence Ranking', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax.tick_params(colors=COLORS[5])
    
    for i, (bar, val) in enumerate(zip(bars, feature_divorce_corr.values)):
        ax.text(val, i, f'  {val:.3f}', va='center', ha='left',
               color=COLORS[5], fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}feature_network_graph.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def probability_density_maps(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0, 0]
    
    comm_range = np.linspace(1, 10, 50)
    conflict_range = np.linspace(0, 9, 50)
    comm_grid, conflict_grid = np.meshgrid(comm_range, conflict_range)
    
    divorce_prob = np.zeros_like(comm_grid)
    
    for i in range(len(conflict_range)):
        for j in range(len(comm_range)):
            subset = df[
                (df['communication_score'] >= comm_range[j] - 0.5) &
                (df['communication_score'] < comm_range[j] + 0.5) &
                (df['conflict_frequency'] >= conflict_range[i] - 0.5) &
                (df['conflict_frequency'] < conflict_range[i] + 0.5)
            ]
            if len(subset) > 10:
                divorce_prob[i, j] = subset['divorced'].mean()
            else:
                divorce_prob[i, j] = np.nan
    
    im = ax.contourf(comm_grid, conflict_grid, divorce_prob, levels=15,
                     cmap='RdYlGn_r', alpha=0.8)
    contour_lines = ax.contour(comm_grid, conflict_grid, divorce_prob, 
                               levels=[0.3, 0.5, 0.7], colors=COLORS[5],
                               linewidths=2, linestyles=['--', '-', '--'])
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%0.1f')
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Communication Score', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Conflict Frequency', fontsize=12, color=COLORS[5])
    ax.set_title('Divorce Probability Landscape', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Divorce Probability', color=COLORS[5], fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[0, 1]
    
    income_range = np.linspace(df['combined_income'].min(), df['combined_income'].max(), 50)
    duration_range = np.linspace(0, 40, 50)
    income_grid, duration_grid = np.meshgrid(income_range, duration_range)
    
    divorce_prob_2 = np.zeros_like(income_grid)
    
    for i in range(len(duration_range)):
        for j in range(len(income_range)):
            subset = df[
                (df['combined_income'] >= income_range[j] - 2000) &
                (df['combined_income'] < income_range[j] + 2000) &
                (df['marriage_duration_years'] >= duration_range[i] - 2) &
                (df['marriage_duration_years'] < duration_range[i] + 2)
            ]
            if len(subset) > 5:
                divorce_prob_2[i, j] = subset['divorced'].mean()
            else:
                divorce_prob_2[i, j] = np.nan
    
    im = ax.contourf(income_grid, duration_grid, divorce_prob_2, levels=15,
                     cmap='coolwarm', alpha=0.8)
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Combined Income ($)', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Marriage Duration (Years)', fontsize=12, color=COLORS[5])
    ax.set_title('Economic-Temporal Risk Map', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Divorce Probability', color=COLORS[5], fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[1, 0]
    
    divorced = df[df['divorced'] == 1]
    married = df[df['divorced'] == 0]
    
    kde_divorced = gaussian_kde(np.vstack([divorced['communication_score'], 
                                           divorced['conflict_frequency']]))
    kde_married = gaussian_kde(np.vstack([married['communication_score'],
                                         married['conflict_frequency']]))
    
    comm_fine = np.linspace(1, 10, 100)
    conflict_fine = np.linspace(0, 9, 100)
    comm_mesh, conflict_mesh = np.meshgrid(comm_fine, conflict_fine)
    positions = np.vstack([comm_mesh.ravel(), conflict_mesh.ravel()])
    
    density_divorced = kde_divorced(positions).reshape(comm_mesh.shape)
    density_married = kde_married(positions).reshape(comm_mesh.shape)
    
    ax.contour(comm_mesh, conflict_mesh, density_married, levels=5,
              colors=COLORS[4], linewidths=2, linestyles='-', alpha=0.7)
    ax.contour(comm_mesh, conflict_mesh, density_divorced, levels=5,
              colors=COLORS[2], linewidths=2, linestyles='--', alpha=0.7)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor=COLORS[4], linewidth=2, label='Married Density'),
        Patch(facecolor='none', edgecolor=COLORS[2], linewidth=2, linestyle='--', label='Divorced Density')
    ]
    ax.legend(handles=legend_elements, framealpha=0.9, facecolor='#080C0D', 
             edgecolor=COLORS[4], loc='upper right')
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Communication Score', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Conflict Frequency', fontsize=12, color=COLORS[5])
    ax.set_title('Distribution Overlap Analysis', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.tick_params(colors=COLORS[5])
    ax.grid(True, alpha=0.1, color=COLORS[4])
    
    ax = axes[1, 1]
    
    age_range = np.linspace(18, 45, 50)
    children_range = np.arange(0, 7)
    age_grid, children_grid = np.meshgrid(age_range, children_range)
    
    divorce_prob_3 = np.zeros_like(age_grid)
    
    for i in range(len(children_range)):
        for j in range(len(age_range)):
            subset = df[
                (df['age_at_marriage'] >= age_range[j] - 1.5) &
                (df['age_at_marriage'] < age_range[j] + 1.5) &
                (df['num_children'] == children_range[i])
            ]
            if len(subset) > 5:
                divorce_prob_3[i, j] = subset['divorced'].mean()
            else:
                divorce_prob_3[i, j] = np.nan
    
    im = ax.imshow(divorce_prob_3, cmap='plasma', aspect='auto', 
                   extent=[age_range.min(), age_range.max(), -0.5, 6.5],
                   origin='lower', alpha=0.8, interpolation='bilinear')
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Age at Marriage', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Number of Children', fontsize=12, color=COLORS[5])
    ax.set_title('Age-Children Risk Surface', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.tick_params(colors=COLORS[5])
    ax.set_yticks(range(7))
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Divorce Probability', color=COLORS[5], fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}probability_density_maps.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def partial_dependence_analysis(df_encoded):
    fig, axes = plt.subplots(2, 3, figsize=(26, 14))
    fig.patch.set_facecolor('#080C0D')
    
    feature_cols = ['communication_score', 'conflict_frequency', 'marriage_duration_years',
                   'combined_income', 'num_children', 'age_at_marriage']
    
    X = df_encoded[feature_cols]
    y = df_encoded['divorced']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    for idx, (ax, feature) in enumerate(zip(axes.flat, feature_cols)):
        
        feature_range = np.linspace(X[feature].min(), X[feature].max(), 50)
        predictions = []
        
        for value in feature_range:
            X_temp = X.copy()
            X_temp[feature] = value
            pred_prob = rf.predict_proba(X_temp)[:, 1].mean()
            predictions.append(pred_prob)
        
        ax.plot(feature_range, predictions, linewidth=3, color=COLORS[5],
               marker='o', markersize=5, markevery=5,
               markeredgecolor=COLORS[4], markeredgewidth=2)
        ax.fill_between(feature_range, predictions, alpha=0.3, color=COLORS[4])
        
        baseline = y.mean()
        ax.axhline(y=baseline, color=COLORS[2], linestyle='--', linewidth=2,
                  alpha=0.6, label=f'Baseline: {baseline:.2%}')
        
        ax.set_facecolor('#0D1214')
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11, color=COLORS[5])
        ax.set_ylabel('Divorce Probability', fontsize=11, color=COLORS[5])
        ax.set_title(f'Effect of {feature.replace("_", " ").title()}', 
                    fontsize=13, color=COLORS[5], pad=15, weight='bold')
        ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], fontsize=9)
        ax.grid(True, alpha=0.15, color=COLORS[4])
        ax.tick_params(colors=COLORS[5])
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}partial_dependence_analysis.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def feature_contribution_waterfall(df_encoded):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor('#080C0D')
    
    feature_cols = ['communication_score', 'conflict_frequency', 'marriage_duration_years',
                   'combined_income', 'num_children', 'age_at_marriage']
    
    X = df_encoded[feature_cols]
    y = df_encoded['divorced']
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    
    ax = axes[0]
    
    baseline = y.mean()
    
    sample_divorced = df_encoded[df_encoded['divorced'] == 1].iloc[0]
    sample_values = sample_divorced[feature_cols]
    
    contributions = []
    cumulative = baseline
    
    for feature in feature_cols:
        X_temp = X.copy()
        X_temp[feature] = sample_values[feature]
        
        pred_with = rf.predict_proba(X_temp)[:, 1].mean()
        contribution = pred_with - cumulative
        contributions.append(contribution)
        cumulative = pred_with
    
    contributions_sorted = sorted(zip(feature_cols, contributions), 
                                 key=lambda x: abs(x[1]), reverse=True)
    features_sorted = [f for f, _ in contributions_sorted]
    values_sorted = [v for _, v in contributions_sorted]
    
    y_pos = 0
    current = baseline
    positions = [y_pos]
    heights = [baseline]
    colors_list = [COLORS[4]]
    
    for i, (feature, contrib) in enumerate(zip(features_sorted, values_sorted)):
        y_pos += 1
        positions.append(y_pos)
        heights.append(contrib)
        colors_list.append(COLORS[2] if contrib > 0 else COLORS[5])
        current += contrib
    
    y_pos += 1
    positions.append(y_pos)
    heights.append(current)
    colors_list.append(COLORS[4])
    
    bottoms = [0]
    running = baseline
    for h in heights[1:-1]:
        bottoms.append(running)
        running += h
    bottoms.append(0)
    
    bars = ax.bar(positions, heights, bottom=bottoms, color=colors_list,
                  edgecolor=COLORS[5], linewidth=2, alpha=0.85, width=0.6)
    
    ax.set_xticks(positions)
    labels = ['Baseline'] + [f.replace('_', '\n').title() for f in features_sorted] + ['Final']
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, color=COLORS[5])
    ax.set_ylabel('Divorce Probability', fontsize=12, color=COLORS[5])
    ax.set_title('High-Risk Case: Feature Contributions', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    for i, (pos, height, bottom) in enumerate(zip(positions[1:-1], heights[1:-1], bottoms[1:-1])):
        y_text = bottom + height/2
        ax.text(pos, y_text, f'{height:+.3f}', ha='center', va='center',
               color=COLORS[0] if abs(height) > 0.1 else COLORS[5], 
               fontsize=9, weight='bold')
    
    ax = axes[1]
    
    sample_married = df_encoded[df_encoded['divorced'] == 0].iloc[0]
    sample_values_2 = sample_married[feature_cols]
    
    contributions_2 = []
    cumulative_2 = baseline
    
    for feature in feature_cols:
        X_temp = X.copy()
        X_temp[feature] = sample_values_2[feature]
        
        pred_with = rf.predict_proba(X_temp)[:, 1].mean()
        contribution = pred_with - cumulative_2
        contributions_2.append(contribution)
        cumulative_2 = pred_with
    
    contributions_sorted_2 = sorted(zip(feature_cols, contributions_2),
                                   key=lambda x: abs(x[1]), reverse=True)
    features_sorted_2 = [f for f, _ in contributions_sorted_2]
    values_sorted_2 = [v for _, v in contributions_sorted_2]
    
    y_pos = 0
    current = baseline
    positions = [y_pos]
    heights = [baseline]
    colors_list = [COLORS[4]]
    
    for i, (feature, contrib) in enumerate(zip(features_sorted_2, values_sorted_2)):
        y_pos += 1
        positions.append(y_pos)
        heights.append(contrib)
        colors_list.append(COLORS[2] if contrib > 0 else COLORS[5])
        current += contrib
    
    y_pos += 1
    positions.append(y_pos)
    heights.append(current)
    colors_list.append(COLORS[4])
    
    bottoms = [0]
    running = baseline
    for h in heights[1:-1]:
        bottoms.append(running)
        running += h
    bottoms.append(0)
    
    bars = ax.bar(positions, heights, bottom=bottoms, color=colors_list,
                  edgecolor=COLORS[5], linewidth=2, alpha=0.85, width=0.6)
    
    ax.set_xticks(positions)
    labels = ['Baseline'] + [f.replace('_', '\n').title() for f in features_sorted_2] + ['Final']
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, color=COLORS[5])
    ax.set_ylabel('Divorce Probability', fontsize=12, color=COLORS[5])
    ax.set_title('Low-Risk Case: Feature Contributions', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    for i, (pos, height, bottom) in enumerate(zip(positions[1:-1], heights[1:-1], bottoms[1:-1])):
        y_text = bottom + height/2
        ax.text(pos, y_text, f'{height:+.3f}', ha='center', va='center',
               color=COLORS[0] if abs(height) > 0.1 else COLORS[5],
               fontsize=9, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}feature_contribution_waterfall.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def main():
    print("Loading divorce dataset for advanced visualization")
    
    df, df_encoded = load_prepare()
    print(f"Analyzing {len(df):,} couples with network and probability mapping techniques")
    
    print("Building feature relationship network")
    feature_network_graph(df)
    
    print("Generating probability density maps")
    probability_density_maps(df)
    
    print("Computing partial dependence curves")
    partial_dependence_analysis(df_encoded)
    
    print("Creating feature contribution waterfalls")
    feature_contribution_waterfall(df_encoded)
    
    print(f"\nAnalysis complete: 4 visualization suites deployed to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
