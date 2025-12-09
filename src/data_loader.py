"""
Módulo para carregamento e preparação dos dados do Titanic
"""

import pandas as pd
import numpy as np
from pathlib import Path

class TitanicDataLoader:
    """Classe para carregar e realizar operações básicas nos dados do Titanic"""
    
    def __init__(self, data_path='../data/raw/'):
        """
        Inicializa o carregador de dados
        
        Args:
            data_path: Caminho para o diretório com os dados brutos
        """
        self.data_path = Path(data_path)
        self.train_df = None
        self.test_df = None
        
    def load_data(self):
        """Carrega os datasets de treino e teste"""
        try:
            self.train_df = pd.read_csv(self.data_path / 'train.csv')
            self.test_df = pd.read_csv(self.data_path / 'test.csv')
            print(f"✓ Dados carregados com sucesso!")
            print(f"  - Treino: {self.train_df.shape}")
            print(f"  - Teste: {self.test_df.shape}")
            return self.train_df, self.test_df
        except FileNotFoundError:
            print("❌ Erro: Arquivos não encontrados!")
            print("   Baixe os dados de: https://www.kaggle.com/c/titanic/data")
            print(f"   E coloque em: {self.data_path}")
            return None, None
    
    def get_feature_types(self, df=None):
        """
        Identifica tipos de features
        
        Returns:
            dict com listas de features numéricas e categóricas
        """
        if df is None:
            df = self.train_df
            
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove identificadores
        if 'PassengerId' in numeric_features:
            numeric_features.remove('PassengerId')
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features
        }
    
    def get_missing_summary(self, df=None):
        """
        Retorna resumo de valores ausentes
        
        Returns:
            DataFrame com informações de missing values
        """
        if df is None:
            df = self.train_df
            
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2)
        })
        
        return missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
    
    def create_basic_features(self, df):
        """
        Cria features básicas de engenharia
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame com novas features
        """
        df_copy = df.copy()
        
        # Tamanho da família
        df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
        
        # Está sozinho?
        df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
        
        # Tem cabine?
        df_copy['HasCabin'] = (~df_copy['Cabin'].isnull()).astype(int)
        
        # Extrai título do nome
        if 'Name' in df_copy.columns:
            df_copy['Title'] = df_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            
            # Agrupa títulos raros
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
                'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
                'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
            }
            df_copy['Title'] = df_copy['Title'].map(title_mapping).fillna('Rare')
        
        # Faixa etária (se Age disponível)
        if 'Age' in df_copy.columns:
            df_copy['AgeGroup'] = pd.cut(
                df_copy['Age'], 
                bins=[0, 12, 18, 60, 100],
                labels=['Child', 'Teen', 'Adult', 'Senior']
            )
        
        return df_copy
    
    def get_data_summary(self):
        """Retorna resumo completo dos dados"""
        if self.train_df is None:
            print("❌ Carregue os dados primeiro com load_data()")
            return None
        
        summary = {
            'shape': self.train_df.shape,
            'columns': self.train_df.columns.tolist(),
            'dtypes': self.train_df.dtypes.to_dict(),
            'missing': self.get_missing_summary(),
            'target_distribution': self.train_df['Survived'].value_counts().to_dict(),
            'feature_types': self.get_feature_types()
        }
        
        return summary


# Funções auxiliares standalone
def load_titanic_data(train_path='../data/raw/train.csv', 
                      test_path='../data/raw/test.csv'):
    """
    Função simples para carregar dados
    
    Returns:
        tuple: (train_df, test_df)
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None


def print_data_info(df, title="Dataset Info"):
    """Imprime informações básicas do dataset"""
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    print(f"\nShape: {df.shape}")
    print(f"\nColunas:\n{df.columns.tolist()}")
    print(f"\nTipos de dados:\n{df.dtypes}")
    print(f"\nValores ausentes:\n{df.isnull().sum()}")
    print("=" * 80)


if __name__ == "__main__":
    # Teste do módulo
    loader = TitanicDataLoader()
    train, test = loader.load_data()
    
    if train is not None:
        print("\n" + "="*80)
        print("RESUMO DOS DADOS".center(80))
        print("="*80)
        summary = loader.get_data_summary()
        print(f"\nShape: {summary['shape']}")
        print(f"\nFeatures numéricas: {summary['feature_types']['numeric']}")
        print(f"\nFeatures categóricas: {summary['feature_types']['categorical']}")
        print(f"\nDistribuição do target: {summary['target_distribution']}")