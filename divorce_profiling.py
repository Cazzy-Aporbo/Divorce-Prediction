import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#0F1618', '#000000', '#4A4682', '#3A5C60', '#4A696E', '#8FC7B8']
GRADIENT = ['#0F1618', '#1F2628', '#2F3638', '#3A5C60', '#4A696E', '#5A7A7E', '#6A8A8E', '#7AAA9E', '#8FC7B8']
OUTPUT_PATH = '/Users/cazandraaporbo/Desktop/mygit/Divorce/'

plt.style.use('dark_background')

class RelationshipProfiler:
    def __init__(self, df):
        self.raw = df.copy()
        self.profiles = None
        self.generate_profiles()
        
    def generate_profiles(self):
        feature_matrix = self.raw[['communication_score', 'conflict_frequency', 
                                   'marriage_duration_years', 'combined_income',
                                   'num_children']].copy()
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.raw['relationship_profile'] = kmeans.fit_predict(scaled)
        
        profile_names = []
        for i in range(5):
            cluster_data = self.raw[self.raw['relationship_profile'] == i]
            avg_comm = cluster_data['communication_score'].mean()
            avg_conflict = cluster_data['conflict_frequency'].mean()
            
            if avg_comm > 7 and avg_conflict < 2:
                name = 'Harmonious'
            elif avg_comm < 5 and avg_conflict > 3:
                name = 'Distressed'
            elif avg_comm > 6 and avg_conflict > 2:
                name = 'Passionate'
            elif avg_comm < 6 and avg_conflict < 2:
                name = 'Disconnected'
            else:
                name = 'Moderate'
                
            profile_names.append(name)
        
        self.raw['profile_name'] = self.raw['relationship_profile'].map(
            {i: profile_names[i] for i in range(5)}
        )
        
        self.profiles = {
            'centroids': kmeans.cluster_centers_,
            'names': profile_names,
            'scaler': scaler
        }

def load_profile_data():
    df = pd.read_csv('/Users/cazandraaporbo/Desktop/mygit/Divorce/divorce_df.csv')
    profiler = RelationshipProfiler(df)
    return profiler.raw, profiler

def survival_analysis_curves(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0, 0]
    
    duration_range = np.arange(0, 41, 1)
    
    divorced_couples = df[df['divorced'] == 1]
    married_couples = df[df['divorced'] == 0]
    
    survival_rates = []
    for duration in duration_range:
        still_married = len(df[df['marriage_duration_years'] >= duration])
        total_at_risk = len(df[df['marriage_duration_years'] >= duration]) + \
                       len(divorced_couples[divorced_couples['marriage_duration_years'] < duration])
        if total_at_risk > 0:
            survival_rates.append(still_married / total_at_risk)
        else:
            survival_rates.append(0)
    
    ax.plot(duration_range, survival_rates, linewidth=4, color=COLORS[5],
           marker='o', markersize=5, markevery=5,
           markeredgecolor=COLORS[4], markeredgewidth=2, label='Overall Survival')
    
    ax.fill_between(duration_range, survival_rates, alpha=0.3, color=COLORS[4])
    
    ax.axhline(y=0.5, color=COLORS[2], linestyle='--', linewidth=2, alpha=0.6, label='50% Threshold')
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Marriage Duration (Years)', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Survival Probability', fontsize=12, color=COLORS[5])
    ax.set_title('Marriage Survival Curve', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    ax.set_ylim(0, 1)
    
    ax = axes[0, 1]
    
    comm_groups = {
        'High (8-10)': df[df['communication_score'] >= 8],
        'Medium (5-7)': df[(df['communication_score'] >= 5) & (df['communication_score'] < 8)],
        'Low (1-4)': df[df['communication_score'] < 5]
    }
    
    for i, (label, group) in enumerate(comm_groups.items()):
        group_divorced = group[group['divorced'] == 1]
        survival_rates_group = []
        
        for duration in duration_range:
            still_married = len(group[group['marriage_duration_years'] >= duration])
            total_at_risk = len(group[group['marriage_duration_years'] >= duration]) + \
                           len(group_divorced[group_divorced['marriage_duration_years'] < duration])
            if total_at_risk > 0:
                survival_rates_group.append(still_married / total_at_risk)
            else:
                survival_rates_group.append(0)
        
        ax.plot(duration_range, survival_rates_group, linewidth=3, 
               color=GRADIENT[i*3], marker='o', markersize=4, markevery=5,
               label=label, markeredgecolor=COLORS[4], markeredgewidth=1.5)
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Marriage Duration (Years)', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Survival Probability', fontsize=12, color=COLORS[5])
    ax.set_title('Survival by Communication Quality', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    ax.set_ylim(0, 1)
    
    ax = axes[1, 0]
    
    income_quartiles = pd.qcut(df['combined_income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        group = df[income_quartiles == quartile]
        group_divorced = group[group['divorced'] == 1]
        survival_rates_group = []
        
        for duration in duration_range:
            still_married = len(group[group['marriage_duration_years'] >= duration])
            total_at_risk = len(group[group['marriage_duration_years'] >= duration]) + \
                           len(group_divorced[group_divorced['marriage_duration_years'] < duration])
            if total_at_risk > 0:
                survival_rates_group.append(still_married / total_at_risk)
            else:
                survival_rates_group.append(0)
        
        ax.plot(duration_range, survival_rates_group, linewidth=3,
               color=GRADIENT[i*2], marker='o', markersize=4, markevery=5,
               label=f'Income {quartile}', markeredgecolor=COLORS[4], markeredgewidth=1.5)
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Marriage Duration (Years)', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Survival Probability', fontsize=12, color=COLORS[5])
    ax.set_title('Survival by Income Quartile', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    ax.set_ylim(0, 1)
    
    ax = axes[1, 1]
    
    hazard_rates = []
    for i in range(len(duration_range) - 1):
        at_risk = len(df[df['marriage_duration_years'] >= duration_range[i]])
        events = len(divorced_couples[(divorced_couples['marriage_duration_years'] >= duration_range[i]) & 
                                     (divorced_couples['marriage_duration_years'] < duration_range[i+1])])
        if at_risk > 0:
            hazard_rates.append(events / at_risk)
        else:
            hazard_rates.append(0)
    
    ax.bar(duration_range[:-1], hazard_rates, width=0.8,
          color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=1.5)
    
    window = 3
    if len(hazard_rates) >= window:
        smoothed = pd.Series(hazard_rates).rolling(window=window, center=True).mean()
        ax.plot(duration_range[:-1], smoothed, linewidth=3, color=COLORS[2],
               label=f'{window}-Year Moving Average', marker='o', markersize=5,
               markeredgecolor=COLORS[5], markeredgewidth=1.5)
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Marriage Duration (Years)', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Hazard Rate', fontsize=12, color=COLORS[5])
    ax.set_title('Divorce Hazard Function', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}survival_analysis_curves.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def risk_profiling_matrix(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0, 0]
    
    profile_divorce = df.groupby('profile_name')['divorced'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    
    bars = ax.barh(range(len(profile_divorce)), profile_divorce['mean'] * 100,
                   color=[GRADIENT[i*2] for i in range(len(profile_divorce))],
                   edgecolor=COLORS[5], linewidth=2, alpha=0.85)
    
    ax.set_yticks(range(len(profile_divorce)))
    ax.set_yticklabels(profile_divorce.index, fontsize=12, color=COLORS[5])
    ax.set_xlabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_title('Relationship Profile Risk Levels', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax.tick_params(colors=COLORS[5])
    
    for i, (bar, val) in enumerate(zip(bars, profile_divorce['mean'] * 100)):
        count = profile_divorce['count'].iloc[i]
        ax.text(val, i, f'  {val:.1f}% (n={count})', va='center', ha='left',
               color=COLORS[5], fontsize=11, weight='bold')
    
    ax = axes[0, 1]
    
    profile_features = df.groupby('profile_name').agg({
        'communication_score': 'mean',
        'conflict_frequency': 'mean',
        'marriage_duration_years': 'mean',
        'combined_income': 'mean'
    })
    
    profile_features_normalized = (profile_features - profile_features.min()) / (profile_features.max() - profile_features.min())
    
    im = ax.imshow(profile_features_normalized.T, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(profile_features.index)))
    ax.set_yticks(range(len(profile_features.columns)))
    ax.set_xticklabels(profile_features.index, rotation=45, ha='right', fontsize=10, color=COLORS[5])
    ax.set_yticklabels(['Communication', 'Conflict', 'Duration', 'Income'],
                       fontsize=11, color=COLORS[5])
    ax.set_title('Profile Feature Signatures', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    
    for i in range(len(profile_features.columns)):
        for j in range(len(profile_features.index)):
            val = profile_features_normalized.T.values[i, j]
            text_color = COLORS[0] if val > 0.5 else COLORS[5]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=text_color, fontsize=10, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[1, 0]
    
    for profile in df['profile_name'].unique():
        profile_data = df[df['profile_name'] == profile]
        
        if len(profile_data) > 50:
            kde_comm = gaussian_kde(profile_data['communication_score'])
            x_range = np.linspace(1, 10, 100)
            density = kde_comm(x_range)
            
            ax.plot(x_range, density, linewidth=3, label=profile,
                   alpha=0.8)
            ax.fill_between(x_range, density, alpha=0.2)
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Communication Score', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Density', fontsize=12, color=COLORS[5])
    ax.set_title('Communication Distribution by Profile', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], fontsize=9)
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 1]
    
    risk_factors = pd.DataFrame({
        'Young Marriage': [df[df['age_at_marriage'] < 25]['divorced'].mean() * 100],
        'Low Income': [df[df['combined_income'] < 40000]['divorced'].mean() * 100],
        'High Conflict': [df[df['conflict_frequency'] > 3]['divorced'].mean() * 100],
        'Low Communication': [df[df['communication_score'] < 5]['divorced'].mean() * 100],
        'Cultural Mismatch': [df[df['cultural_background_match'] == 0]['divorced'].mean() * 100],
        'Many Children': [df[df['num_children'] >= 3]['divorced'].mean() * 100]
    })
    
    baseline = df['divorced'].mean() * 100
    
    relative_risk = risk_factors.iloc[0] / baseline
    
    bars = ax.barh(range(len(risk_factors.columns)), risk_factors.iloc[0].values,
                   color=[GRADIENT[i] for i in range(len(risk_factors.columns))],
                   edgecolor=COLORS[5], linewidth=2, alpha=0.85)
    
    ax.axvline(x=baseline, color=COLORS[2], linestyle='--', linewidth=2, alpha=0.7,
              label=f'Baseline: {baseline:.1f}%')
    
    ax.set_yticks(range(len(risk_factors.columns)))
    ax.set_yticklabels(risk_factors.columns, fontsize=11, color=COLORS[5])
    ax.set_xlabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_title('Risk Factor Analysis', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax.tick_params(colors=COLORS[5])
    
    for i, (bar, val) in enumerate(zip(bars, risk_factors.iloc[0].values)):
        rr = relative_risk.iloc[i]
        ax.text(val, i, f'  {val:.1f}% (RR={rr:.2f})', va='center', ha='left',
               color=COLORS[5], fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}risk_profiling_matrix.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def temporal_pattern_analysis(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0, 0]
    
    year_bins = pd.cut(df['marriage_duration_years'], bins=range(0, 41, 2))
    divorce_by_year = df.groupby(year_bins)['divorced'].mean() * 100
    counts_by_year = df.groupby(year_bins).size()
    
    x = range(len(divorce_by_year))
    
    ax2 = ax.twinx()
    
    bars = ax.bar(x, divorce_by_year.values, color=COLORS[4], alpha=0.7,
                  edgecolor=COLORS[5], linewidth=1.5)
    line = ax2.plot(x, counts_by_year.values, color=COLORS[5], linewidth=3,
                   marker='o', markersize=7, markeredgecolor=COLORS[4],
                   markeredgewidth=2, label='Sample Size')
    
    ax.set_xticks(range(0, len(divorce_by_year), 2))
    ax.set_xticklabels([f'{i*2}-{i*2+2}' for i in range(0, len(divorce_by_year), 2)],
                       rotation=45, ha='right', fontsize=9, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_xlabel('Marriage Duration (Years)', fontsize=12, color=COLORS[5])
    ax2.set_ylabel('Number of Couples', fontsize=12, color=COLORS[5])
    ax.set_title('Temporal Divorce Pattern', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax2.set_facecolor('#0D1214')
    ax.tick_params(colors=COLORS[5])
    ax2.tick_params(colors=COLORS[5])
    ax2.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    
    ax = axes[0, 1]
    
    age_bins = pd.cut(df['age_at_marriage'], bins=[18, 22, 26, 30, 35, 45])
    
    for i, age_group in enumerate(age_bins.cat.categories):
        group = df[age_bins == age_group]
        duration_bins = pd.cut(group['marriage_duration_years'], bins=range(0, 41, 5))
        divorce_rates = group.groupby(duration_bins)['divorced'].mean() * 100
        
        x_vals = range(len(divorce_rates))
        ax.plot(x_vals, divorce_rates.values, linewidth=2.5, marker='o',
               markersize=6, label=str(age_group), color=GRADIENT[i],
               markeredgecolor=COLORS[4], markeredgewidth=1.5)
    
    ax.set_xticks(range(len(divorce_rates)))
    ax.set_xticklabels(['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40'],
                       rotation=45, ha='right', fontsize=9, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_xlabel('Marriage Duration (Years)', fontsize=12, color=COLORS[5])
    ax.set_title('Age-Stratified Temporal Patterns', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], 
             fontsize=9, title='Marriage Age')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 0]
    
    critical_periods = {
        'Honeymoon (0-2y)': df[df['marriage_duration_years'] <= 2]['divorced'].mean() * 100,
        'Early (3-7y)': df[(df['marriage_duration_years'] > 2) & (df['marriage_duration_years'] <= 7)]['divorced'].mean() * 100,
        'Seven Year (7-10y)': df[(df['marriage_duration_years'] > 7) & (df['marriage_duration_years'] <= 10)]['divorced'].mean() * 100,
        'Middle (10-20y)': df[(df['marriage_duration_years'] > 10) & (df['marriage_duration_years'] <= 20)]['divorced'].mean() * 100,
        'Long-term (20+y)': df[df['marriage_duration_years'] > 20]['divorced'].mean() * 100
    }
    
    bars = ax.bar(range(len(critical_periods)), list(critical_periods.values()),
                  color=[GRADIENT[i] for i in range(len(critical_periods))],
                  edgecolor=COLORS[5], linewidth=2, alpha=0.85)
    
    ax.set_xticks(range(len(critical_periods)))
    ax.set_xticklabels(list(critical_periods.keys()), rotation=45, ha='right',
                       fontsize=11, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_title('Critical Period Analysis', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    for i, (bar, val) in enumerate(zip(bars, critical_periods.values())):
        ax.text(i, val, f'{val:.1f}%', ha='center', va='bottom',
               color=COLORS[5], fontsize=11, weight='bold')
    
    ax = axes[1, 1]
    
    duration_quartiles = pd.qcut(df['marriage_duration_years'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    factors = ['communication_score', 'conflict_frequency', 'combined_income', 'num_children']
    
    correlations = []
    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        quartile_data = df[duration_quartiles == quartile]
        corr_values = []
        for factor in factors:
            corr = np.corrcoef(quartile_data[factor], quartile_data['divorced'])[0, 1]
            corr_values.append(abs(corr))
        correlations.append(corr_values)
    
    correlations = np.array(correlations).T
    
    im = ax.imshow(correlations, cmap='plasma', aspect='auto')
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(len(factors)))
    ax.set_xticklabels(['Q1\n(0-25%)', 'Q2\n(25-50%)', 'Q3\n(50-75%)', 'Q4\n(75-100%)'],
                       fontsize=10, color=COLORS[5])
    ax.set_yticklabels(['Communication', 'Conflict', 'Income', 'Children'],
                       fontsize=11, color=COLORS[5])
    ax.set_xlabel('Duration Quartile', fontsize=12, color=COLORS[5])
    ax.set_title('Factor Importance Over Time', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    
    for i in range(len(factors)):
        for j in range(4):
            val = correlations[i, j]
            text_color = COLORS[0] if val > 0.3 else COLORS[5]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color=text_color, fontsize=10, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Correlation|', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}temporal_pattern_analysis.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def multidimensional_relationship_space(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0, 0]
    
    divorced = df[df['divorced'] == 1]
    married = df[df['divorced'] == 0]
    
    ax.scatter(married['communication_score'], married['conflict_frequency'],
              c=married['marriage_duration_years'], cmap='Blues', s=30, alpha=0.4,
              edgecolors='none', label='Married')
    
    scatter = ax.scatter(divorced['communication_score'], divorced['conflict_frequency'],
                        c=divorced['marriage_duration_years'], cmap='Reds', s=30, alpha=0.6,
                        edgecolors=COLORS[4], linewidths=0.5, label='Divorced')
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Communication Score', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Conflict Frequency', fontsize=12, color=COLORS[5])
    ax.set_title('Relationship Space Topology', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Marriage Duration', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[0, 1]
    
    for profile in df['profile_name'].unique():
        profile_data = df[df['profile_name'] == profile]
        
        ax.scatter(profile_data['combined_income'], profile_data['marriage_duration_years'],
                  s=50, alpha=0.6, edgecolors=COLORS[4], linewidths=0.5,
                  label=profile)
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Combined Income', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Marriage Duration (Years)', fontsize=12, color=COLORS[5])
    ax.set_title('Profile Distribution in Socioeconomic Space', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], fontsize=9)
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 0]
    
    stability_score = df['communication_score'] - df['conflict_frequency']
    
    divorced_stability = stability_score[df['divorced'] == 1]
    married_stability = stability_score[df['divorced'] == 0]
    
    bins = np.linspace(stability_score.min(), stability_score.max(), 30)
    
    ax.hist(married_stability, bins=bins, alpha=0.6, color=COLORS[4],
           edgecolor=COLORS[5], linewidth=1.5, label='Married')
    ax.hist(divorced_stability, bins=bins, alpha=0.6, color=COLORS[2],
           edgecolor=COLORS[5], linewidth=1.5, label='Divorced')
    
    ax.axvline(x=married_stability.mean(), color=COLORS[4], linestyle='--',
              linewidth=2, label=f'Married Mean: {married_stability.mean():.2f}')
    ax.axvline(x=divorced_stability.mean(), color=COLORS[2], linestyle='--',
              linewidth=2, label=f'Divorced Mean: {divorced_stability.mean():.2f}')
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Stability Score (Comm - Conflict)', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Frequency', fontsize=12, color=COLORS[5])
    ax.set_title('Relationship Stability Distribution', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], fontsize=9)
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 1]
    
    profile_counts = df.groupby(['profile_name', 'divorced']).size().unstack(fill_value=0)
    profile_counts_pct = profile_counts.div(profile_counts.sum(axis=1), axis=0) * 100
    
    x = np.arange(len(profile_counts_pct))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, profile_counts_pct[0], width, label='Stayed Married',
                   color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
    bars2 = ax.bar(x + width/2, profile_counts_pct[1], width, label='Divorced',
                   color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(profile_counts_pct.index, rotation=45, ha='right',
                       fontsize=11, color=COLORS[5])
    ax.set_ylabel('Percentage', fontsize=12, color=COLORS[5])
    ax.set_title('Outcome Distribution by Profile', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}multidimensional_relationship_space.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def main():
    print("Loading relationship profiling system")
    
    df, profiler = load_profile_data()
    print(f"Profiled {len(df):,} couples into {df['profile_name'].nunique()} relationship archetypes")
    
    print("Generating survival analysis curves")
    survival_analysis_curves(df)
    
    print("Building risk profiling matrix")
    risk_profiling_matrix(df)
    
    print("Analyzing temporal patterns")
    temporal_pattern_analysis(df)
    
    print("Mapping multidimensional relationship space")
    multidimensional_relationship_space(df)
    
    print(f"\nAnalysis complete: 4 visualization suites deployed to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
