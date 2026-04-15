"""
treinar_modelo.py
─────────────────────────────────────────────────────────────────────────────
Script de treinamento offline da Árvore de Decisão usada no ESP32.
Gera os thresholds que são copiados para o firmware (.ino).

Fluxo:
  1. Carrega dataset de amostras rotuladas (CSV)
  2. Treina DecisionTreeClassifier (max_depth=3 → compatível com firmware)
  3. Avalia acurácia e exporta thresholds para o código C

Dependências: scikit-learn, pandas, matplotlib
  pip install scikit-learn pandas matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ─── 1. Geração de dataset sintético (substitua por dados reais) ─────────────
np.random.seed(42)
N = 500

temp = np.concatenate([
    np.random.uniform(18, 29, 300),   # normal
    np.random.uniform(29, 37, 120),   # alerta
    np.random.uniform(37, 45, 80),    # crítico
])
hum = np.concatenate([
    np.random.uniform(35, 75, 300),
    np.random.uniform(20, 35,  60) if True else [],
    np.concatenate([np.random.uniform(20, 35, 60), np.random.uniform(80, 95, 60)]),
    np.random.uniform(30, 70,  80),
])[:N]
aq = np.concatenate([
    np.random.uniform(100, 395, 300),
    np.random.uniform(395, 695, 120),
    np.random.uniform(695, 900,  80),
])

# Rótulos baseados nos thresholds do firmware
def rotular(t, h, q):
    if q >= 700 or t >= 38:
        return 2  # CRÍTICO
    if q >= 400 or t >= 30 or h < 30 or h > 80:
        return 1  # ALERTA
    return 0      # NORMAL

rotulos = np.array([rotular(t, h, q) for t, h, q in zip(temp, hum, aq)])

df = pd.DataFrame({'temp': temp, 'hum': hum, 'aq': aq, 'estado': rotulos})
df.to_csv('dataset_ambiental.csv', index=False)
print(f"Dataset gerado: {len(df)} amostras\n{df['estado'].value_counts().to_dict()}")

# ─── 2. Treino ───────────────────────────────────────────────────────────────
X = df[['temp', 'hum', 'aq']].values
y = df['estado'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
clf.fit(X_train, y_train)

# ─── 3. Avaliação ────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n── Relatório de Classificação ──")
print(classification_report(y_test, y_pred,
      target_names=['NORMAL', 'ALERTA', 'CRITICO']))

acc = (y_pred == y_test).mean() * 100
print(f"Acurácia no conjunto de teste: {acc:.1f}%")

# ─── 4. Estrutura da árvore (para extrair thresholds) ────────────────────────
print("\n── Estrutura da Árvore ──")
feature_names = ['temp', 'hum', 'aq']
print(export_text(clf, feature_names=feature_names))

# ─── 5. Exportar thresholds para C ──────────────────────────────────────────
thresholds = clf.tree_.threshold
features   = clf.tree_.feature

print("\n── Thresholds extraídos (copiar para .ino) ──")
for i, (f, t) in enumerate(zip(features, thresholds)):
    if f >= 0:  # -2 = folha
        print(f"  Nó {i}: {feature_names[f]} >= {t:.1f}")

# ─── 6. Visualização ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cores = {0: 'green', 1: 'orange', 2: 'red'}
nomes = {0: 'NORMAL', 1: 'ALERTA', 2: 'CRÍTICO'}

for ax, (xi, xj, xl, yl) in zip(axes, [
    (temp, hum,  'Temperatura (°C)', 'Umidade (%)'),
    (temp, aq,   'Temperatura (°C)', 'Qualidade do Ar (ppm)'),
    (hum,  aq,   'Umidade (%)',      'Qualidade do Ar (ppm)'),
]):
    for cls in [0, 1, 2]:
        mask = rotulos == cls
        ax.scatter(xi[mask], xj[mask], c=cores[cls], label=nomes[cls],
                   alpha=0.5, s=15)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.legend(fontsize=7)

plt.suptitle('Distribuição das Classes – Dataset Ambiental', fontsize=12)
plt.tight_layout()
plt.savefig('distribuicao_classes.png', dpi=150)
print("\nGráfico salvo em distribuicao_classes.png")
