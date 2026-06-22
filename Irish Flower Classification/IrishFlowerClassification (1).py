
"""
TASK 1: Iris Flower Classification
===================================
Classifies Iris flowers (setosa, versicolor, virginica) using multiple ML models.
Dataset: Built-in sklearn Iris dataset (same data from UCI/codealpha)
"""
#Md. Miraj-Ul-Islam, Data science Intern at CodeAlpha, 2026

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['target'] = iris.target

print("=" * 55)
print("     IRIS FLOWER CLASSIFICATION REPORT")
print("=" * 55)
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"🌸 Classes: {list(iris.target_names)}")
print(f"\n--- First 5 Rows ---")
print(df.head().to_string())
print(f"\n--- Dataset Statistics ---")
print(df.describe().round(2).to_string())
print(f"\n--- Class Distribution ---")
print(df['species'].value_counts().to_string())
print(f"\n--- Missing Values ---")
print(df.isnull().sum().to_string())

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────
X = df[iris.feature_names]
y = df['target']

# Add derived features
df['petal_area']  = df['petal length (cm)']  * df['petal width (cm)']
df['sepal_area']  = df['sepal length (cm)']  * df['sepal width (cm)']
df['petal_ratio'] = df['petal length (cm)']  / (df['petal width (cm)'] + 1e-6)
df['sepal_ratio'] = df['sepal length (cm)']  / (df['sepal width (cm)'] + 1e-6)

X_enhanced = df[list(iris.feature_names) + ['petal_area','sepal_area',
                                              'petal_ratio','sepal_ratio']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n--- Train/Test Split ---")
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# 3. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
models = {
    'Logistic Regression'    : LogisticRegression(max_iter=200, random_state=42),
    'K-Nearest Neighbors'    : KNeighborsClassifier(n_neighbors=5),
    'Decision Tree'          : DecisionTreeClassifier(random_state=42, max_depth=4),
    'Random Forest'          : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)'              : SVC(kernel='rbf', C=1.0, random_state=42),
    'Gradient Boosting'      : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results    = {}
cv_scores  = {}

print("\n--- Model Training & Evaluation ---\n")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred  = model.predict(X_test_s)
    acc     = accuracy_score(y_test, y_pred)
    cv_sc   = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
    results[name]   = {'model': model, 'y_pred': y_pred, 'accuracy': acc}
    cv_scores[name] = cv_sc
    print(f"  {name:<26}  Test Acc: {acc*100:.1f}%  |  CV: {cv_sc.mean()*100:.1f}% ± {cv_sc.std()*100:.1f}%")

best_name  = max(results, key=lambda n: results[n]['accuracy'])
best_model = results[best_name]['model']
best_pred  = results[best_name]['y_pred']
print(f"\n🏆 Best Model: {best_name}  ({results[best_name]['accuracy']*100:.1f}%)")

print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, best_pred, target_names=iris.target_names))

# ─────────────────────────────────────────────
# 4. VISUALIZATIONS  (10-panel figure)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

COLORS  = ['#FF6B6B', '#4ECDC4', '#45B7D1']
SPECIES = list(iris.target_names)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# ── Panel 1: Feature Distributions (4-in-1 violin) ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
feat = 'petal length (cm)'
for i, sp in enumerate(SPECIES):
    data = df[df['species'] == sp][feat]
    parts = ax1.violinplot([data], positions=[i], widths=0.6, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS[i]); pc.set_alpha(0.7)
    parts['cmedians'].set_color('white')
ax1.set_xticks([0,1,2]); ax1.set_xticklabels(SPECIES, fontsize=8)
style_ax(ax1, 'Petal Length Distribution', ylabel='cm')

# ── Panel 2: Petal Width violin ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
feat2 = 'petal width (cm)'
for i, sp in enumerate(SPECIES):
    data = df[df['species'] == sp][feat2]
    parts = ax2.violinplot([data], positions=[i], widths=0.6, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS[i]); pc.set_alpha(0.7)
    parts['cmedians'].set_color('white')
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(SPECIES, fontsize=8)
style_ax(ax2, 'Petal Width Distribution', ylabel='cm')

# ── Panel 3: Sepal scatter ────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
for i, sp in enumerate(SPECIES):
    mask = df['species'] == sp
    ax3.scatter(df[mask]['sepal length (cm)'], df[mask]['sepal width (cm)'],
                c=COLORS[i], label=sp, alpha=0.75, s=55, edgecolors='white', linewidth=0.4)
ax3.legend(fontsize=8, facecolor='#161b22', labelcolor='white', framealpha=0.6)
style_ax(ax3, 'Sepal: Length vs Width', 'Sepal Length (cm)', 'Sepal Width (cm)')

# ── Panel 4: Petal scatter ────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for i, sp in enumerate(SPECIES):
    mask = df['species'] == sp
    ax4.scatter(df[mask]['petal length (cm)'], df[mask]['petal width (cm)'],
                c=COLORS[i], label=sp, alpha=0.75, s=55, edgecolors='white', linewidth=0.4)
ax4.legend(fontsize=8, facecolor='#161b22', labelcolor='white', framealpha=0.6)
style_ax(ax4, 'Petal: Length vs Width', 'Petal Length (cm)', 'Petal Width (cm)')

# ── Panel 5: Confusion Matrix (best model) ───────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
cm  = confusion_matrix(y_test, best_pred)
im  = ax5.imshow(cm, cmap='Blues', aspect='auto')
for i in range(3):
    for j in range(3):
        ax5.text(j, i, cm[i,j], ha='center', va='center',
                 color='white' if cm[i,j] > cm.max()/2 else '#8b949e', fontsize=13, fontweight='bold')
ax5.set_xticks([0,1,2]); ax5.set_xticklabels(SPECIES, fontsize=8, rotation=15)
ax5.set_yticks([0,1,2]); ax5.set_yticklabels(SPECIES, fontsize=8)
style_ax(ax5, f'Confusion Matrix\n({best_name})', 'Predicted', 'Actual')
plt.colorbar(im, ax=ax5, fraction=0.04)

# ── Panel 6: Model Accuracy Comparison ───────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
names_short = [n.replace(' ', '\n') for n in results.keys()]
accs        = [v['accuracy']*100 for v in results.values()]
bar_colors  = ['#FFD700' if n == best_name else '#4ECDC4' for n in results.keys()]
bars = ax6.barh(names_short, accs, color=bar_colors, edgecolor='#30363d', height=0.6)
for bar, acc in zip(bars, accs):
    ax6.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2,
             f'{acc:.1f}%', va='center', ha='right', color='#0d1117', fontsize=8, fontweight='bold')
ax6.set_xlim(85, 102)
style_ax(ax6, 'Model Accuracy Comparison', 'Accuracy (%)')

# ── Panel 7: CV Score Box Plot ────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
cv_data  = [cv_scores[n]*100 for n in results.keys()]
bp = ax7.boxplot(cv_data, patch_artist=True, medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], ['#FF6B6B','#4ECDC4','#45B7D1','#FFD700','#C9A0DC','#FF8C42']):
    patch.set_facecolor(color); patch.set_alpha(0.75)
for element in ['whiskers','caps','fliers']:
    for item in bp[element]: item.set_color('#8b949e')
ax7.set_xticks(range(1, len(results)+1))
ax7.set_xticklabels([n.split()[0] for n in results.keys()], fontsize=7, rotation=20)
style_ax(ax7, '5-Fold Cross-Validation Scores', 'Model', 'CV Accuracy (%)')

# ── Panel 8: Feature Importances (Random Forest) ─────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
rf   = results['Random Forest']['model']
imps = rf.feature_importances_
feat_names = [f.replace(' (cm)', '') for f in iris.feature_names]
sorted_idx = np.argsort(imps)
ax8.barh([feat_names[i] for i in sorted_idx], imps[sorted_idx],
         color=['#FF6B6B','#4ECDC4','#45B7D1','#FFD700'], edgecolor='#30363d')
style_ax(ax8, 'Feature Importances\n(Random Forest)', 'Importance', 'Feature')

# ── Panel 9: Correlation Heatmap ─────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
corr = df[list(iris.feature_names)].corr()
short_names = ['Sep.L','Sep.W','Pet.L','Pet.W']
mask_upper  = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax9, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=short_names, yticklabels=short_names,
            annot_kws={'size': 9, 'color': 'white'}, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax9.set_facecolor('#161b22')
ax9.tick_params(colors='#8b949e', labelsize=8)
ax9.set_title('Feature Correlation Heatmap', color='white', fontsize=11, fontweight='bold', pad=8)

# ────────────────────────────────────────────────
# Main Title
# ────────────────────────────────────────────────
fig.suptitle(
    '🌸 IRIS FLOWER CLASSIFICATION — Complete Analysis',
    color='white',
    fontsize=16,
    fontweight='bold',
    y=0.98
)

# ────────────────────────────────────────────────
# Save Figure
# ────────────────────────────────────────────────
import os

# Create output directory if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# File path
save_path = os.path.join(output_dir, "task1_iris_classification.png")

# Save image
plt.savefig(
    save_path,
    dpi=150,
    bbox_inches='tight',
    facecolor='#0d1117'
)

# Show figure in notebook (optional)
plt.show()

# Close figure
plt.close()

print(f"\n✅ Task 1 complete — chart saved successfully!")
print(f"📁 Location: {os.path.abspath(save_path)}")


#ROC Curve — All Models

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV

y_test_bin = label_binarize(y_test, classes=[0,1,2])
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')

colors_roc = ['#FF6B6B','#4ECDC4','#45B7D1','#FFD700','#C9A0DC','#FF8C42']
for (name, val), col in zip(results.items(), colors_roc):
    model = val['model']
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test_s)
    else:
        y_score = CalibratedClassifierCV(model).fit(X_train_s, y_train).predict_proba(X_test_s)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC={roc_auc:.2f})')

ax.plot([0,1],[0,1], 'w--', lw=1)
ax.set_title('ROC Curve — All Models', color='white', fontweight='bold')
ax.set_xlabel('False Positive Rate', color='#8b949e')
ax.set_ylabel('True Positive Rate', color='#8b949e')
ax.tick_params(colors='#8b949e')
ax.legend(fontsize=8, facecolor='#161b22', labelcolor='white', framealpha=0.6)
for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
plt.tight_layout(); plt.show()



# Some other analysis with visualization


#Precision / Recall / F1 — All Models

from sklearn.metrics import precision_score, recall_score, f1_score

metrics_dict = {'Precision':[], 'Recall':[], 'F1':[]}
names = list(results.keys())

for name in names:
    yp = results[name]['y_pred']
    metrics_dict['Precision'].append(precision_score(y_test, yp, average='weighted'))
    metrics_dict['Recall'].append(recall_score(y_test, yp, average='weighted'))
    metrics_dict['F1'].append(f1_score(y_test, yp, average='weighted'))

x = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(10,5))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')

ax.bar(x - width, metrics_dict['Precision'], width, label='Precision', color='#FF6B6B', edgecolor='#30363d')
ax.bar(x,         metrics_dict['Recall'],    width, label='Recall',    color='#4ECDC4', edgecolor='#30363d')
ax.bar(x + width, metrics_dict['F1'],        width, label='F1-Score',  color='#FFD700', edgecolor='#30363d')

ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ','\n') for n in names], color='#8b949e', fontsize=8)
ax.set_ylim(0.8, 1.02)
ax.set_title('Precision / Recall / F1 — All Models', color='white', fontweight='bold')
ax.set_ylabel('Score', color='#8b949e')
ax.tick_params(colors='#8b949e')
ax.legend(fontsize=9, facecolor='#161b22', labelcolor='white', framealpha=0.6)
for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
plt.tight_layout(); plt.show()


#CV score Heatmap (Model × Fold)

fig, ax = plt.subplots(figsize=(9,4))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')

data_matrix = np.array([cv_scores[n]*100 for n in results.keys()])
im = ax.imshow(data_matrix, cmap='YlGn', aspect='auto', vmin=88, vmax=100)

for i in range(len(results)):
    for j in range(5):
        ax.text(j, i, f'{data_matrix[i,j]:.1f}', ha='center', va='center',
                color='black', fontsize=9, fontweight='bold')

ax.set_xticks(range(5))
ax.set_xticklabels([f'Fold {i+1}' for i in range(5)], color='#8b949e')
ax.set_yticks(range(len(results)))
ax.set_yticklabels(list(results.keys()), color='#8b949e', fontsize=8)
ax.set_title('CV Score Heatmap (Model × Fold)', color='white', fontweight='bold')
ax.tick_params(colors='#8b949e')
plt.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout(); plt.show()


#Learning Curve

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_s, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', random_state=42)

fig, ax = plt.subplots(figsize=(8,5))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')

ax.plot(train_sizes, train_scores.mean(axis=1)*100, 'o-', color='#4ECDC4', label='Training Score')
ax.fill_between(train_sizes,
                (train_scores.mean(1) - train_scores.std(1))*100,
                (train_scores.mean(1) + train_scores.std(1))*100,
                alpha=0.2, color='#4ECDC4')

ax.plot(train_sizes, val_scores.mean(axis=1)*100, 'o-', color='#FF6B6B', label='Validation Score')
ax.fill_between(train_sizes,
                (val_scores.mean(1) - val_scores.std(1))*100,
                (val_scores.mean(1) + val_scores.std(1))*100,
                alpha=0.2, color='#FF6B6B')

ax.set_title(f'Learning Curve — {best_name}', color='white', fontweight='bold')
ax.set_xlabel('Training Samples', color='#8b949e')
ax.set_ylabel('Accuracy (%)', color='#8b949e')
ax.tick_params(colors='#8b949e')
ax.legend(fontsize=9, facecolor='#161b22', labelcolor='white', framealpha=0.6)
for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
plt.tight_layout(); plt.show()


#Prediction Errors per Class per Model

fig, ax = plt.subplots(figsize=(9,5))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')

x = np.arange(3)
width = 0.13
colors_err = ['#FF6B6B','#4ECDC4','#45B7D1','#FFD700','#C9A0DC','#FF8C42']

for idx, (name, val) in enumerate(results.items()):
    errors = []
    for cls in range(3):
        mask = y_test == cls
        err = (val['y_pred'][mask] != y_test[mask]).sum()
        errors.append(err)
    ax.bar(x + idx*width, errors, width, label=name,
           color=colors_err[idx], edgecolor='#30363d')

ax.set_xticks(x + width*2.5)
ax.set_xticklabels(iris.target_names, color='#8b949e')
ax.set_title('Prediction Errors per Class per Model', color='white', fontweight='bold')
ax.set_ylabel('Number of Errors', color='#8b949e')
ax.tick_params(colors='#8b949e')
ax.legend(fontsize=7, facecolor='#161b22', labelcolor='white', framealpha=0.6)
for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
plt.tight_layout(); plt.show()


#Class Distribution

fig = plt.figure(figsize=(30, 155))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

def style_ax2(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=8)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=8)
    ax.tick_params(colors='#8b949e', labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')

# ── Panel 1: Class Distribution Bar ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
class_counts = df['species'].value_counts()
bars = ax1.bar(class_counts.index, class_counts.values,
               color=COLORS, edgecolor='#30363d', width=0.5)
for bar, val in zip(bars, class_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha='center', color='white', fontsize=10, fontweight='bold')
ax1.set_ylim(0, 65)
style_ax2(ax1, 'Class Distribution', 'Species', 'Count')
plt.show()



