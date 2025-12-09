# 📊 Análise Exploratória de Dados - Titanic
---
## 📋 O que foi Analisado

### 1️⃣ Compreensão do Conjunto de Dados
- ✅ Estrutura: 891 observações, 12 atributos
- ✅ Tipos de dados: 5 numéricos, 5 categóricos
- ✅ Estatísticas descritivas completas
- ✅ Distribuições (média, mediana, quartis, assimetria)
- ✅ Identificação de outliers
- ✅ Cardinalidade de atributos categóricos
- ✅ Análise de valores ausentes (missing values)

### 2️⃣ Importância e Relacionamentos
- ✅ Matriz de correlação entre atributos numéricos
- ✅ Teste qui-quadrado para atributos categóricos
- ✅ Identificação de preditores fortes
- ✅ Análise de multicolinearidade
- ✅ Sugestões de feature engineering

### 3️⃣ Visualizações Geradas
- ✅ Histogramas de distribuições numéricas
- ✅ Box plots agrupados por sobrevivência
- ✅ Gráficos de barras para categóricos
- ✅ Heatmap de correlação
- ✅ Análise de valores ausentes
- ✅ Distribuição da variável alvo

### 4️⃣ Exploração para Árvore de Decisão
- ✅ Ranking de preditores para splits
- ✅ Identificação de problemas (overfitting/underfitting)
- ✅ Estratégias para classes desbalanceadas
- ✅ Recomendação de métricas de avaliação
- ✅ Sugestão de hiperparâmetros iniciais

---

## 📊 Principais Descobertas

### 🎯 Preditores Mais Importantes

| Atributo | Tipo | Significância | Observação |
|----------|------|---------------|------------|
| **Sex** | Categórico | p < 0.001 | ⭐⭐⭐ Preditor mais forte |
| **Pclass** | Numérico | p < 0.001 | ⭐⭐⭐ Correlação: -0.34 |
| **Fare** | Numérico | r = 0.26 | ⭐⭐ Positiva moderada |
| **Age** | Numérico | Após imputação | ⭐ Importante |
| **Embarked** | Categórico | p < 0.001 | ⭐ Significativo |

### ⚠️ Problemas Identificados

**Valores Ausentes:**
- `Age`: 177 valores (19.9%)
- `Cabin`: 687 valores (77.1%)
- `Embarked`: 2 valores (0.2%)

**Qualidade dos Dados:**
- Distribuição assimétrica em `Fare` (skewness = 4.79)
- Outliers presentes em `Age` e `Fare`
- Alta cardinalidade em `Cabin` (147 valores únicos)
- `Name` e `Ticket` são únicos (requerem extração)

**Desbalanceamento de Classes:**
- Não sobreviveu: 549 (62%)
- Sobreviveu: 342 (38%)
- Razão: 1.62:1 (moderadamente desbalanceado)

---

## 💡 Recomendações para Pré-processamento

### 1. Imputação de Valores Ausentes

```python
# Age - Imputar por grupo (melhor performance)
df['Age'].fillna(
    df.groupby(['Pclass', 'Sex'])['Age'].transform('median'), 
    inplace=True
)

# Cabin - Criar flag binária
df['HasCabin'] = df['Cabin'].notna().astype(int)

# Embarked - Imputar pela moda
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

### 2. Feature Engineering Sugerida

```python
# Tamanho da família
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Viajando sozinho
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Título extraído do nome
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Agrupar títulos raros: Mr, Mrs, Miss, Master, Rare

# Deck da cabine (primeira letra)
df['Deck'] = df['Cabin'].str[0]

# Faixas etárias
df['AgeGroup'] = pd.cut(df['Age'], 
                        bins=[0, 12, 18, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Senior'])

# Tarifa por pessoa
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
```

### 3. Tratamento de Outliers (Opcional)

```python
# Fare - Considerar log transform ou cap nos percentis
df['Fare_log'] = np.log1p(df['Fare'])

# Ou limitar outliers extremos
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df['Fare_capped'] = df['Fare'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)
```

---

## 🌳 Recomendações para Modelagem

### Hiperparâmetros Iniciais (Ponto de Partida)

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,                    # Evitar overfitting (dataset pequeno)
    min_samples_split=20,           # ~2% dos dados
    min_samples_leaf=10,            # ~1% dos dados
    max_leaf_nodes=20,              # Limitar complexidade
    class_weight='balanced',        # Compensar desbalanceamento
    criterion='gini',               # Testar também 'entropy'
    random_state=42
)
```

### Hiperparâmetros para Experimentar

| Parâmetro | Valores Sugeridos | Impacto |
|-----------|-------------------|---------|
| `max_depth` | [3, 5, 7, 10, None] | Controla overfitting |
| `min_samples_split` | [10, 20, 30, 50] | Mínimo para dividir nó |
| `min_samples_leaf` | [5, 10, 15, 20] | Mínimo em folha |
| `criterion` | ['gini', 'entropy'] | Método de split |
| `max_features` | [None, 'sqrt', 'log2'] | Features por split |

### Métricas de Avaliação Recomendadas

```python
from sklearn.metrics import (
    f1_score,           # ⭐ PRINCIPAL (classes desbalanceadas)
    roc_auc_score,      # ⭐ ROBUSTA
    confusion_matrix,   # ⭐ OBRIGATÓRIA
    classification_report,
    accuracy_score
)

# Cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1-Score médio: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**⚠️ NÃO confiar apenas em Acurácia!** (pode ser enganosa com desbalanceamento)

---

## 📁 Arquivos Gerados

### Visualizações
Todas as figuras foram salvas em: `../results/figures/`

1. `01_histogramas_numericos.png` - Distribuições
2. `02_boxplots_por_target.png` - Comparação por sobrevivência
3. `03_analise_Sex.png` - Análise categórica
4. `03_analise_Pclass.png` - Análise de classe
5. `03_analise_Embarked.png` - Porto de embarque
6. `04_heatmap_correlacao.png` - Matriz de correlação
7. `05_distribuicao_target.png` - Variável alvo
8. `06_valores_ausentes.png` - Missing values

### Dados Exportados
- `../results/eda_results.pkl` - Resultados da análise em formato pickle

**Conteúdo do arquivo pickle:**
```python
import pickle

with open('../results/eda_results.pkl', 'rb') as f:
    eda = pickle.load(f)

# Disponível:
eda['numeric_features']              # Lista de features numéricas
eda['categorical_features']          # Lista de features categóricas
eda['missing_summary']               # DataFrame com missing values
eda['correlation_matrix']            # Matriz de correlação
eda['chi2_results']                  # Resultados dos testes qui-quadrado
eda['feature_importance_ranking']    # Ranking de importância
eda['recommended_hyperparameters']   # Hiperparâmetros sugeridos
eda['recommended_metrics']           # Métricas recomendadas
```

---

## 🚀 Próximos Passos

### Etapa 2: Pré-processamento (`02_preprocessing.ipynb`)

**O que fazer:**
1. ✅ Carregar os dados originais
2. ✅ Implementar imputação conforme recomendado
3. ✅ Criar features engineered sugeridas
4. ✅ Codificar variáveis categóricas (One-Hot ou Label Encoding)
5. ✅ Normalizar/padronizar se necessário
6. ✅ Dividir em treino/validação/teste
7. ✅ Salvar dados processados

**Usar como base:**
- Estratégias de imputação documentadas acima
- Features sugeridas na análise
- Módulo `src/data_loader.py`

### Etapa 3: Modelagem (`03_decision_tree_model.ipynb`)

**O que fazer:**
1. ✅ Carregar dados processados
2. ✅ Treinar DecisionTreeClassifier com hiperparâmetros iniciais
3. ✅ Avaliar com F1-Score, AUC-ROC e Matriz de Confusão
4. ✅ Fazer Grid Search ou Random Search
5. ✅ Comparar com Random Forest
6. ✅ Visualizar árvore resultante
7. ✅ Analisar feature importance
8. ✅ Validação cruzada (5-fold)
9. ✅ Fazer predições no conjunto de teste
10. ✅ Documentar resultados

**Usar como base:**
- Hiperparâmetros iniciais recomendados
- Métricas de avaliação definidas
- Features selecionadas como importantes

---

## 📚 Módulos Auxiliares Criados

### `src/data_loader.py`
Funções para carregar e manipular dados:
```python
from src.data_loader import TitanicDataLoader

loader = TitanicDataLoader()
train, test = loader.load_data()
features = loader.get_feature_types()
missing = loader.get_missing_summary()
```

### `src/visualization.py`
Funções para criar visualizações:
```python
from src.visualization import TitanicVisualizer

viz = TitanicVisualizer()
viz.plot_correlation_heatmap(df, numeric_cols)
viz.plot_categorical_analysis(df, 'Sex')
```

---

## 🔍 Como Reproduzir esta Análise

### 1. Baixar os dados
```bash
# Acesse: https://www.kaggle.com/c/titanic/data
# Baixe train.csv e test.csv
# Coloque em: data/raw/
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Executar o notebook
```bash
cd notebooks
jupyter notebook 01_exploratory_analysis.ipynb
```
---

## 📊 Estatísticas Rápidas

```
Dataset: 891 observações × 12 atributos

Target (Survived):
├─ Não (0): 549 passageiros (61.6%)
└─ Sim  (1): 342 passageiros (38.4%)

Missing Values:
├─ Age:      177 (19.9%) ⚠️
├─ Cabin:    687 (77.1%) ⚠️⚠️
└─ Embarked:   2 ( 0.2%)

Correlações com Survived:
├─ Pclass: -0.34 (negativa) ⭐⭐
├─ Fare:   +0.26 (positiva) ⭐⭐
└─ Age:    -0.08 (fraca)

Testes Qui-Quadrado:
├─ Sex:      p < 0.001 ⭐⭐⭐
├─ Pclass:   p < 0.001 ⭐⭐⭐
└─ Embarked: p < 0.001 ⭐⭐
```

---

## ❓ Perguntas Frequentes

### P: Posso usar outras features além das sugeridas?
**R:** Sim! As sugestões são um ponto de partida. Experimente criar outras combinações.

### P: Preciso seguir exatamente os hiperparâmetros recomendados?
**R:** Não. São valores iniciais baseados na análise. Experimente outros valores!

### P: E se eu quiser testar outros algoritmos?
**R:** Ótimo! A análise serve de base para qualquer modelo. Compare os resultados!

### P: Como cito esta análise na apresentação?
**R:** Exemplo: *"A análise exploratória identificou Sex e Pclass como os preditores mais significativos (p < 0.001), orientando nossa estratégia de feature selection..."*

---