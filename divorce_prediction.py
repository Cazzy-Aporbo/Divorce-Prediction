import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import chi2_contingency, pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#0F1618', '#000000', '#4A4682', '#3A5C60', '#4A696E', '#8FC7B8']
GRADIENT = ['#0F1618', '#1F2628', '#2F3638', '#3A5C60', '#4A696E', '#5A7A7E', '#6A8A8E', '#7AAA9E', '#8FC7B8']
OUTPUT_PATH = '/Users/cazandraaporbo/Desktop/mygit/Divorce/'

plt.style.use('dark_background')

def load_engineer_features(filepath):
    df = pd.read_csv(filepath)
    
    df['income_per_child'] = df['combined_income'] / (df['num_children'] + 1)
    df['marriage_stage'] = pd.cut(df['marriage_duration_years'], 
                                   bins=[0, 5, 15, 50], 
                                   labels=['Early', 'Middle', 'Long'])
    df['conflict_comm_ratio'] = df['conflict_frequency'] / (df['communication_score'] + 0.1)
    df['age_gap_proxy'] = df['age_at_marriage'].apply(lambda x: 1 if x < 22 or x > 35 else 0)
    df['high_conflict'] = (df['conflict_frequency'] > df['conflict_frequency'].median()).astype(int)
    df['low_communication'] = (df['communication_score'] < df['communication_score'].median()).astype(int)
    df['stability_score'] = (df['communication_score'] - df['conflict_frequency']).clip(lower=-10, upper=10)
    
    return df

def predictive_power_analysis(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0, 0]
    
    numeric_cols = ['age_at_marriage', 'marriage_duration_years', 'num_children', 
                   'combined_income', 'communication_score', 'conflict_frequency']
    
    corr_with_divorce = []
    for col in numeric_cols:
        corr, _ = pearsonr(df[col], df['divorced'])
        corr_with_divorce.append(abs(corr))
    
    bars = ax.barh(range(len(numeric_cols)), corr_with_divorce,
                   color=[GRADIENT[i % len(GRADIENT)] for i in range(len(numeric_cols))],
                   edgecolor=COLORS[5], linewidth=2, alpha=0.85)
    
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels([col.replace('_', ' ').title() for col in numeric_cols], 
                       fontsize=11, color=COLORS[5])
    ax.set_xlabel('Absolute Correlation with Divorce', fontsize=12, color=COLORS[5])
    ax.set_title('Feature Predictive Strength', fontsize=16, color=COLORS[5], 
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax.tick_params(colors=COLORS[5])
    
    for i, (bar, val) in enumerate(zip(bars, corr_with_divorce)):
        ax.text(val, i, f'  {val:.3f}', va='center', ha='left', 
               color=COLORS[5], fontsize=10, weight='bold')
    
    ax = axes[0, 1]
    
    le = LabelEncoder()
    df_encoded = df.copy()
    for col in ['education_level', 'employment_status', 'religious_compatibility']:
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
    
    feature_cols = ['communication_score', 'conflict_frequency', 'marriage_duration_years',
                   'combined_income', 'num_children', 'education_level_encoded',
                   'employment_status_encoded', 'cultural_background_match']
    
    X = df_encoded[feature_cols]
    y = df_encoded['divorced']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    bars = ax.barh(range(len(feature_cols)), importances[indices],
                   color=[GRADIENT[i % len(GRADIENT)] for i in range(len(feature_cols))],
                   edgecolor=COLORS[5], linewidth=2, alpha=0.85)
    
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i].replace('_', ' ').title() for i in indices],
                       fontsize=10, color=COLORS[5])
    ax.set_xlabel('Random Forest Importance', fontsize=12, color=COLORS[5])
    ax.set_title('Machine Learning Feature Ranking', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 0]
    
    comm_bins = pd.qcut(df['communication_score'], q=5, duplicates='drop')
    conflict_bins = pd.qcut(df['conflict_frequency'], q=5, duplicates='drop')
    
    contingency = pd.crosstab(comm_bins, df['divorced'])
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
    
    x = np.arange(len(contingency_pct.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, contingency_pct[0], width,
                   label='Stayed Married', color=COLORS[4], alpha=0.85,
                   edgecolor=COLORS[5], linewidth=2)
    bars2 = ax.bar(x + width/2, contingency_pct[1], width,
                   label='Divorced', color=COLORS[2], alpha=0.85,
                   edgecolor=COLORS[5], linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{i+1}' for i in range(len(contingency_pct.index))],
                       fontsize=11, color=COLORS[5])
    ax.set_ylabel('Percentage', fontsize=12, color=COLORS[5])
    ax.set_xlabel('Communication Score Quintile', fontsize=12, color=COLORS[5])
    ax.set_title('Communication Impact on Divorce Rate', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 1]
    
    duration_bins = pd.cut(df['marriage_duration_years'], bins=[0, 5, 10, 20, 50],
                          labels=['0-5y', '5-10y', '10-20y', '20+y'])
    divorce_by_duration = df.groupby(duration_bins)['divorced'].agg(['mean', 'count'])
    
    x = np.arange(len(divorce_by_duration.index))
    
    ax2 = ax.twinx()
    
    bars = ax.bar(x, divorce_by_duration['mean'] * 100, 
                  color=COLORS[4], alpha=0.7, edgecolor=COLORS[5], linewidth=2)
    line = ax2.plot(x, divorce_by_duration['count'], color=COLORS[5], 
                   linewidth=3, marker='o', markersize=10,
                   markeredgecolor=COLORS[4], markeredgewidth=2, label='Sample Size')
    
    ax.set_xticks(x)
    ax.set_xticklabels(divorce_by_duration.index, fontsize=11, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_xlabel('Marriage Duration', fontsize=12, color=COLORS[5])
    ax2.set_ylabel('Number of Couples', fontsize=12, color=COLORS[5])
    ax.set_title('Temporal Divorce Risk Profile', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax2.set_facecolor('#0D1214')
    ax.tick_params(colors=COLORS[5])
    ax2.tick_params(colors=COLORS[5])
    ax2.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}predictive_power_analysis.png', dpi=300, 
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def classification_performance(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#080C0D')
    
    le = LabelEncoder()
    df_encoded = df.copy()
    for col in ['education_level', 'employment_status', 'religious_compatibility', 'marriage_stage']:
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
    
    feature_cols = ['age_at_marriage', 'marriage_duration_years', 'num_children',
                   'combined_income', 'communication_score', 'conflict_frequency',
                   'cultural_background_match', 'education_level_encoded',
                   'employment_status_encoded', 'religious_compatibility_encoded',
                   'conflict_comm_ratio', 'stability_score']
    
    X = df_encoded[feature_cols]
    y = df_encoded['divorced']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    ax = axes[0, 0]
    
    for i, (name, model) in enumerate(models.items()):
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=3, label=f'{name} (AUC = {roc_auc:.3f})',
               color=GRADIENT[i*3], marker='o', markersize=4, markevery=20,
               markeredgecolor=COLORS[4], markeredgewidth=1.5)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random Guess')
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('False Positive Rate', fontsize=12, color=COLORS[5])
    ax.set_ylabel('True Positive Rate', fontsize=12, color=COLORS[5])
    ax.set_title('ROC Curve Comparison', fontsize=16, color=COLORS[5], 
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], loc='lower right')
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[0, 1]
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', alpha=0.8)
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                          ha='center', va='center', color=COLORS[5] if cm_normalized[i, j] < 0.5 else COLORS[0],
                          fontsize=14, weight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: Stay', 'Predicted: Divorce'], fontsize=11, color=COLORS[5])
    ax.set_yticklabels(['Actual: Stay', 'Actual: Divorce'], fontsize=11, color=COLORS[5])
    ax.set_title('Confusion Matrix: Random Forest', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Rate', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[1, 0]
    
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    
    plot_tree(dt, feature_names=feature_cols, class_names=['Stay', 'Divorce'],
             filled=True, ax=ax, fontsize=9, rounded=True,
             impurity=False, proportion=True)
    
    ax.set_title('Decision Tree: Divorce Prediction Logic', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    
    ax = axes[1, 1]
    
    precision_scores = []
    recall_scores = []
    thresholds_range = np.linspace(0.2, 0.8, 30)
    
    rf_model.fit(X_train, y_train)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    for threshold in thresholds_range:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_thresh)
        
        if cm.sum() > 0 and cm[1, 1] + cm[0, 1] > 0 and cm[1, 1] + cm[1, 0] > 0:
            precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
            recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            precision_scores.append(precision)
            recall_scores.append(recall)
        else:
            precision_scores.append(0)
            recall_scores.append(0)
    
    ax.plot(thresholds_range, precision_scores, linewidth=3, color=COLORS[4],
           marker='o', markersize=6, label='Precision',
           markeredgecolor=COLORS[5], markeredgewidth=1.5)
    ax.plot(thresholds_range, recall_scores, linewidth=3, color=COLORS[2],
           marker='s', markersize=6, label='Recall',
           markeredgecolor=COLORS[5], markeredgewidth=1.5)
    
    f1_scores = 2 * (np.array(precision_scores) * np.array(recall_scores)) / (np.array(precision_scores) + np.array(recall_scores) + 1e-10)
    ax.plot(thresholds_range, f1_scores, linewidth=3, color=COLORS[5],
           marker='D', markersize=6, label='F1 Score',
           markeredgecolor=COLORS[4], markeredgewidth=1.5, linestyle='--')
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Classification Threshold', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Score', fontsize=12, color=COLORS[5])
    ax.set_title('Threshold Optimization Analysis', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}classification_performance.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def relationship_dynamics_matrix(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0, 0]
    
    scatter = ax.scatter(df['communication_score'], df['conflict_frequency'],
                        c=df['divorced'], cmap='coolwarm', s=50, alpha=0.6,
                        edgecolors=COLORS[4], linewidths=0.5)
    
    divorced = df[df['divorced'] == 1]
    married = df[df['divorced'] == 0]
    
    ax.scatter(married['communication_score'].mean(), married['conflict_frequency'].mean(),
              s=500, c=COLORS[4], marker='X', edgecolors=COLORS[5], linewidths=3,
              label='Married Centroid', zorder=10)
    ax.scatter(divorced['communication_score'].mean(), divorced['conflict_frequency'].mean(),
              s=500, c=COLORS[2], marker='X', edgecolors=COLORS[5], linewidths=3,
              label='Divorced Centroid', zorder=10)
    
    ax.set_facecolor('#0D1214')
    ax.set_xlabel('Communication Score', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Conflict Frequency', fontsize=12, color=COLORS[5])
    ax.set_title('Communication-Conflict Phase Space', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Divorced', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[0, 1]
    
    religious_divorce = df.groupby('religious_compatibility')['divorced'].agg(['mean', 'count'])
    religious_divorce = religious_divorce.sort_values('mean', ascending=False)
    
    bars = ax.barh(range(len(religious_divorce)), religious_divorce['mean'] * 100,
                   color=[GRADIENT[i*2] for i in range(len(religious_divorce))],
                   edgecolor=COLORS[5], linewidth=2, alpha=0.85)
    
    ax.set_yticks(range(len(religious_divorce)))
    ax.set_yticklabels(religious_divorce.index, fontsize=11, color=COLORS[5])
    ax.set_xlabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_title('Religious Compatibility Impact', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
    ax.tick_params(colors=COLORS[5])
    
    for i, (bar, val) in enumerate(zip(bars, religious_divorce['mean'] * 100)):
        count = religious_divorce['count'].iloc[i]
        ax.text(val, i, f'  {val:.1f}% (n={count})', va='center', ha='left',
               color=COLORS[5], fontsize=10, weight='bold')
    
    ax = axes[1, 0]
    
    income_bins = pd.qcut(df['combined_income'], q=5, duplicates='drop')
    children_groups = df['num_children'].clip(upper=3)
    
    divorce_rates = df.groupby([income_bins, children_groups])['divorced'].mean().unstack()
    
    im = ax.imshow(divorce_rates.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(divorce_rates.shape[1]))
    ax.set_yticks(range(divorce_rates.shape[0]))
    ax.set_xticklabels([f'{int(x)} kids' for x in divorce_rates.columns], 
                       fontsize=10, color=COLORS[5])
    ax.set_yticklabels([f'Q{i+1}' for i in range(len(divorce_rates.index))],
                       fontsize=10, color=COLORS[5])
    ax.set_xlabel('Number of Children', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Income Quintile', fontsize=12, color=COLORS[5])
    ax.set_title('Income-Children Divorce Risk Heatmap', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    
    for i in range(divorce_rates.shape[0]):
        for j in range(divorce_rates.shape[1]):
            val = divorce_rates.values[i, j]
            if not np.isnan(val):
                text_color = COLORS[0] if val > 0.5 else COLORS[5]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=10, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Divorce Rate', color=COLORS[5])
    cbar.ax.yaxis.set_tick_params(color=COLORS[5])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])
    
    ax = axes[1, 1]
    
    education_order = ['No Formal Education', 'High School', 'Bachelor', 'Master', 'PhD']
    education_divorce = df.groupby('education_level')['divorced'].mean().reindex(education_order) * 100
    education_counts = df.groupby('education_level').size().reindex(education_order)
    
    x = np.arange(len(education_divorce))
    
    ax2 = ax.twinx()
    
    bars = ax.bar(x, education_divorce.values, color=COLORS[4], alpha=0.7,
                  edgecolor=COLORS[5], linewidth=2)
    line = ax2.plot(x, education_counts.values, color=COLORS[5], linewidth=3,
                   marker='o', markersize=10, markeredgecolor=COLORS[4],
                   markeredgewidth=2, label='Sample Size')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['None', 'HS', 'Bach', 'Mast', 'PhD'],
                       fontsize=11, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_xlabel('Education Level', fontsize=12, color=COLORS[5])
    ax2.set_ylabel('Number of Couples', fontsize=12, color=COLORS[5])
    ax.set_title('Education Level vs Divorce Probability', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax2.set_facecolor('#0D1214')
    ax.tick_params(colors=COLORS[5])
    ax2.tick_params(colors=COLORS[5])
    ax2.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}relationship_dynamics_matrix.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def interaction_effects_analysis(df):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#080C0D')
    
    ax = axes[0, 0]
    
    young_marriage = df[df['age_at_marriage'] < 25]
    older_marriage = df[df['age_at_marriage'] >= 25]
    
    young_by_duration = young_marriage.groupby(pd.cut(young_marriage['marriage_duration_years'], 
                                                       bins=[0, 5, 10, 20, 50]))['divorced'].mean() * 100
    older_by_duration = older_marriage.groupby(pd.cut(older_marriage['marriage_duration_years'],
                                                       bins=[0, 5, 10, 20, 50]))['divorced'].mean() * 100
    
    x = np.arange(len(young_by_duration))
    width = 0.35
    
    ax.bar(x - width/2, young_by_duration.values, width, label='Married < 25',
          color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
    ax.bar(x + width/2, older_by_duration.values, width, label='Married ≥ 25',
          color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['0-5y', '5-10y', '10-20y', '20+y'], fontsize=11, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_xlabel('Marriage Duration', fontsize=12, color=COLORS[5])
    ax.set_title('Age at Marriage × Duration Interaction', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[0, 1]
    
    high_comm = df[df['communication_score'] >= 7]
    low_comm = df[df['communication_score'] < 7]
    
    conflict_bins = [0, 1, 2, 3, 10]
    high_comm_conflict = high_comm.groupby(pd.cut(high_comm['conflict_frequency'], 
                                                   bins=conflict_bins))['divorced'].mean() * 100
    low_comm_conflict = low_comm.groupby(pd.cut(low_comm['conflict_frequency'],
                                                 bins=conflict_bins))['divorced'].mean() * 100
    
    x_vals = np.arange(len(high_comm_conflict))
    
    ax.plot(x_vals, high_comm_conflict.values, linewidth=3, color=COLORS[5],
           marker='o', markersize=8, label='High Communication (≥7)',
           markeredgecolor=COLORS[4], markeredgewidth=2)
    ax.plot(x_vals, low_comm_conflict.values, linewidth=3, color=COLORS[2],
           marker='s', markersize=8, label='Low Communication (<7)',
           markeredgecolor=COLORS[4], markeredgewidth=2)
    
    ax.set_xticks(x_vals)
    ax.set_xticklabels(['0-1', '1-2', '2-3', '3+'], fontsize=11, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_xlabel('Monthly Conflicts', fontsize=12, color=COLORS[5])
    ax.set_title('Communication × Conflict Interaction', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 0]
    
    low_income = df[df['combined_income'] < df['combined_income'].median()]
    high_income = df[df['combined_income'] >= df['combined_income'].median()]
    
    children_range = range(0, 5)
    low_income_children = [low_income[low_income['num_children'] == n]['divorced'].mean() * 100 
                          for n in children_range]
    high_income_children = [high_income[high_income['num_children'] == n]['divorced'].mean() * 100
                           for n in children_range]
    
    ax.plot(children_range, low_income_children, linewidth=3, color=COLORS[2],
           marker='o', markersize=8, label='Below Median Income',
           markeredgecolor=COLORS[4], markeredgewidth=2)
    ax.plot(children_range, high_income_children, linewidth=3, color=COLORS[5],
           marker='s', markersize=8, label='Above Median Income',
           markeredgecolor=COLORS[4], markeredgewidth=2)
    
    ax.set_xlabel('Number of Children', fontsize=12, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_title('Income × Children Interaction', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4])
    ax.tick_params(colors=COLORS[5])
    
    ax = axes[1, 1]
    
    cultural_match = df[df['cultural_background_match'] == 1]
    cultural_diff = df[df['cultural_background_match'] == 0]
    
    religious_categories = ['Same Religion', 'Different Religion', 'Not Religious']
    match_rates = [cultural_match[cultural_match['religious_compatibility'] == cat]['divorced'].mean() * 100
                   for cat in religious_categories]
    diff_rates = [cultural_diff[cultural_diff['religious_compatibility'] == cat]['divorced'].mean() * 100
                  for cat in religious_categories]
    
    x = np.arange(len(religious_categories))
    width = 0.35
    
    ax.bar(x - width/2, match_rates, width, label='Cultural Match',
          color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
    ax.bar(x + width/2, diff_rates, width, label='Cultural Difference',
          color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Same\nReligion', 'Different\nReligion', 'Not\nReligious'],
                       fontsize=10, color=COLORS[5])
    ax.set_ylabel('Divorce Rate (%)', fontsize=12, color=COLORS[5])
    ax.set_title('Cultural × Religious Compatibility', fontsize=16, color=COLORS[5],
                pad=20, weight='bold')
    ax.set_facecolor('#0D1214')
    ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
    ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
    ax.tick_params(colors=COLORS[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}interaction_effects_analysis.png', dpi=300,
                facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
    plt.close()

def main():
    print("Loading divorce prediction dataset")
    
    df = load_engineer_features('/Users/cazandraaporbo/Desktop/mygit/Divorce/divorce_df.csv')
    print(f"Analyzing {len(df):,} couples across {len(df.columns)} features")
    
    print("Generating predictive power analysis")
    predictive_power_analysis(df)
    
    print("Building classification models")
    classification_performance(df)
    
    print("Mapping relationship dynamics")
    relationship_dynamics_matrix(df)
    
    print("Computing interaction effects")
    interaction_effects_analysis(df)
    
    print(f"\nAnalysis complete: 4 visualization suites deployed to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
