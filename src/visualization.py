"""
Módulo de visualização para análise exploratória do Titanic
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Configurações globais de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TitanicVisualizer:
    """Classe para criar visualizações dos dados do Titanic"""
    
    def __init__(self, save_path='../results/figures/'):
        """
        Inicializa o visualizador
        
        Args:
            save_path: Caminho para salvar as figuras
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def plot_numeric_distributions(self, df, numeric_cols, save=True):
        """
        Cria histogramas para features numéricas
        
        Args:
            df: DataFrame
            numeric_cols: Lista de colunas numéricas
            save: Se True, salva a figura
        """
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        fig.suptitle('Distribuição dos Atributos Numéricos', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            data = df[col].dropna()
            
            # Histograma
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_title(f'{col}\n(n={len(data)}, missing={df[col].isnull().sum()})', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
            
            # Adiciona estatísticas
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.1f}')
            ax.legend(fontsize=8)
        
        # Remove eixos extras
        for idx in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_path / '01_histogramas_numericos.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Figura salva: {filepath}")
        
        return fig
    
    def plot_boxplots_by_target(self, df, numeric_cols, target='Survived', save=True):
        """
        Cria boxplots agrupados pela variável alvo
        
        Args:
            df: DataFrame
            numeric_cols: Lista de colunas numéricas
            target: Nome da coluna alvo
            save: Se True, salva a figura
        """
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        fig.suptitle(f'Box Plots por {target}', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            # Prepara dados
            data = df[[col, target]].dropna()
            
            # Boxplot
            box_parts = data.boxplot(column=col, by=target, ax=ax, patch_artist=True)
            
            ax.set_title(col, fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{target} (0=Não, 1=Sim)')
            ax.set_ylabel(col)
            plt.sca(ax)
            plt.xticks([1, 2], ['Não', 'Sim'])
            ax.get_figure().suptitle('')  # Remove título padrão do pandas
        
        # Remove eixos extras
        for idx in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_path / '02_boxplots_por_target.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Figura salva: {filepath}")
        
        return fig
    
    def plot_categorical_analysis(self, df, cat_col, target='Survived', save=True):
        """
        Cria análise visual de uma feature categórica
        
        Args:
            df: DataFrame
            cat_col: Nome da coluna categórica
            target: Nome da coluna alvo
            save: Se True, salva a figura
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Análise de {cat_col}', fontsize=16, fontweight='bold')
        
        # 1. Frequência geral
        value_counts = df[cat_col].value_counts()
        axes[0].bar(range(len(value_counts)), value_counts.values, 
                    color='skyblue', edgecolor='black')
        axes[0].set_xticks(range(len(value_counts)))
        axes[0].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[0].set_title(f'Frequência de {cat_col}')
        axes[0].set_ylabel('Contagem')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Adiciona valores nas barras
        for i, v in enumerate(value_counts.values):
            axes[0].text(i, v, str(v), ha='center', va='bottom')
        
        # 2. Sobrevivência por categoria (contagem)
        survival_counts = df.groupby([cat_col, target]).size().unstack(fill_value=0)
        survival_counts.plot(kind='bar', ax=axes[1], 
                            color=['coral', 'lightgreen'], edgecolor='black')
        axes[1].set_title(f'Sobrevivência por {cat_col} (Contagem)')
        axes[1].set_ylabel('Contagem')
        axes[1].set_xlabel(cat_col)
        axes[1].legend(['Não Sobreviveu', 'Sobreviveu'], title=target)
        axes[1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Taxa de sobrevivência (%)
        survival_rate = df.groupby(cat_col)[target].mean() * 100
        bars = axes[2].bar(range(len(survival_rate)), survival_rate.values,
                          color='mediumseagreen', edgecolor='black')
        axes[2].set_xticks(range(len(survival_rate)))
        axes[2].set_xticklabels(survival_rate.index, rotation=45, ha='right')
        axes[2].set_title(f'Taxa de Sobrevivência por {cat_col}')
        axes[2].set_ylabel('Taxa de Sobrevivência (%)')
        axes[2].axhline(y=df[target].mean()*100, color='red', linestyle='--', 
                       linewidth=2, label=f'Média Geral: {df[target].mean()*100:.1f}%')
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
        
        # Adiciona valores nas barras
        for i, v in enumerate(survival_rate.values):
            axes[2].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_path / f'03_analise_{cat_col}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Figura salva: {filepath}")
        
        return fig
    
    def plot_correlation_heatmap(self, df, numeric_cols, save=True):
        """
        Cria heatmap de correlação
        
        Args:
            df: DataFrame
            numeric_cols: Lista de colunas numéricas
            save: Se True, salva a figura
        """
        # Calcula matriz de correlação
        corr_matrix = df[numeric_cols].corr()
        
        # Cria figura
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Matriz de Correlação - Atributos Numéricos', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_path / '04_heatmap_correlacao.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Figura salva: {filepath}")
        
        return fig
    
    def plot_target_distribution(self, df, target='Survived', save=True):
        """
        Visualiza distribuição da variável alvo
        
        Args:
            df: DataFrame
            target: Nome da coluna alvo
            save: Se True, salva a figura
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Distribuição da Variável Alvo: {target}', 
                     fontsize=14, fontweight='bold')
        
        # Contagem
        value_counts = df[target].value_counts().sort_index()
        axes[0].bar(value_counts.index, value_counts.values, 
                   color=['coral', 'lightgreen'], edgecolor='black', width=0.6)
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Não Sobreviveu', 'Sobreviveu'])
        axes[0].set_ylabel('Contagem')
        axes[0].set_title('Contagem Absoluta')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Adiciona valores e percentuais
        total = len(df)
        for i, v in enumerate(value_counts.values):
            pct = v / total * 100
            axes[0].text(i, v, f'{v}\n({pct:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
        
        # Pizza
        axes[1].pie(value_counts.values, labels=['Não Sobreviveu', 'Sobreviveu'],
                   autopct='%1.1f%%', startangle=90, colors=['coral', 'lightgreen'],
                   explode=(0.05, 0.05), shadow=True)
        axes[1].set_title('Proporção')
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_path / '05_distribuicao_target.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Figura salva: {filepath}")
        
        return fig
    
    def plot_missing_values(self, df, save=True):
        """
        Visualiza padrão de valores ausentes
        
        Args:
            df: DataFrame
            save: Se True, salva a figura
        """
        # Calcula missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        missing_pct = (missing / len(df) * 100).round(2)
        
        if len(missing) == 0:
            print("✓ Não há valores ausentes para visualizar!")
            return None
        
        # Cria figura
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Análise de Valores Ausentes (Missing)', 
                     fontsize=14, fontweight='bold')
        
        # Gráfico de barras - contagem
        axes[0].barh(range(len(missing)), missing.values, color='indianred', edgecolor='black')
        axes[0].set_yticks(range(len(missing)))
        axes[0].set_yticklabels(missing.index)
        axes[0].set_xlabel('Quantidade de Valores Ausentes')
        axes[0].set_title('Contagem Absoluta')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Adiciona valores
        for i, v in enumerate(missing.values):
            axes[0].text(v, i, f' {v}', va='center')
        
        # Gráfico de barras - percentual
        axes[1].barh(range(len(missing_pct)), missing_pct.values, 
                    color='salmon', edgecolor='black')
        axes[1].set_yticks(range(len(missing_pct)))
        axes[1].set_yticklabels(missing_pct.index)
        axes[1].set_xlabel('Percentual de Valores Ausentes (%)')
        axes[1].set_title('Percentual')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Adiciona valores
        for i, v in enumerate(missing_pct.values):
            axes[1].text(v, i, f' {v}%', va='center')
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_path / '06_valores_ausentes.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Figura salva: {filepath}")
        
        return fig
    
    def create_full_report(self, df, numeric_cols, categorical_cols, target='Survived'):
        """
        Cria relatório visual completo
        
        Args:
            df: DataFrame
            numeric_cols: Lista de colunas numéricas
            categorical_cols: Lista de colunas categóricas
            target: Nome da coluna alvo
        """
        print("\n" + "="*80)
        print("GERANDO RELATÓRIO VISUAL COMPLETO".center(80))
        print("="*80 + "\n")
        
        # 1. Distribuição do target
        self.plot_target_distribution(df, target)
        
        # 2. Valores ausentes
        self.plot_missing_values(df)
        
        # 3. Distribuições numéricas
        self.plot_numeric_distributions(df, numeric_cols)
        
        # 4. Boxplots por target
        self.plot_boxplots_by_target(df, numeric_cols, target)
        
        # 5. Heatmap de correlação
        self.plot_correlation_heatmap(df, numeric_cols + [target])
        
        # 6. Análise categórica
        for cat_col in categorical_cols[:3]:  # Primeiras 3 categóricas
            self.plot_categorical_analysis(df, cat_col, target)
        
        print("\n" + "="*80)
        print("✓ RELATÓRIO COMPLETO GERADO!".center(80))
        print(f"Figuras salvas em: {self.save_path}".center(80))
        print("="*80)


# Funções auxiliares standalone
def quick_plot_distribution(data, title="Distribuição"):
    """Cria gráfico rápido de distribuição"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frequência')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def quick_correlation_plot(df, cols):
    """Cria heatmap rápido de correlação"""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Matriz de Correlação', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Teste do módulo
    print("Módulo de visualização carregado com sucesso!")
    print("\nUso:")
    print("  from visualization import TitanicVisualizer")
    print("  viz = TitanicVisualizer()")
    print("  viz.create_full_report(df, numeric_cols, categorical_cols)")