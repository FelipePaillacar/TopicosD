import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, precision_score, 
    recall_score, f1_score, RocCurveDisplay, PrecisionRecallDisplay
)
# Rutas locales (archivos .csv)
TRAIN_PATH = "KDDTrain+.csv"
TEST_PATH = "KDDTest+.csv"

def load_nsl_kdd_csv(data_path):
    """Carga un archivo .csv del dataset NSL-KDD."""
    if not os.path.exists(data_path):
        print(f"Error: Archivo no encontrado -> {data_path}")
        return None
    return pd.read_csv(data_path)

# Construcción de una función que realice el particionado completo
def train_val_test_split(df, rstate=99, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

# --- Transformadores Personalizados ---

# Transformador creado para elimar las filas con valores nulos
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.dropna()

# Transformador diseñado para escalar de manera sencilla únicamente unas columnas seleccionadas
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        return X_copy

# Transformador para codificar únicamente las columnas categoricas y devolver un DataFrame
class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._oh = OneHotEncoder(sparse=False)
        self._columns = None
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])
        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh = pd.DataFrame(X_cat_oh, 
                                columns=self._columns, 
                                index=X_copy.index)
        X_copy.drop(list(X_cat), axis=1, inplace=True)
        return X_copy.join(X_cat_oh)

# Construcción de un pipeline para los atributos numéricos global
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('rbst_scaler', RobustScaler()),
])

# Transformador para codificar únicamente las columnas categoricas y devolver un df (usando toarray)
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._oh = OneHotEncoder()
        self._columns = None
        
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self
        
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])
        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh = pd.DataFrame(X_cat_oh.toarray(), 
                                columns=self._columns, 
                                index=X_copy.index)
        X_copy.drop(list(X_cat), axis=1, inplace=True)
        return X_copy.join(X_cat_oh)

# Transformador que prepara todo el conjunto de datos llamando pipelines y transformadores personalizados
class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._full_pipeline = None
        self._columns = None
        
    def fit(self, X, y=None):
        num_attribs = list(X.select_dtypes(exclude=['object']))
        cat_attribs = list(X.select_dtypes(include=['object']))
        self._full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", CustomOneHotEncoder(), cat_attribs),
        ])
        self._full_pipeline.fit(X)
        self._columns = pd.get_dummies(X).columns
        return self
        
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_prep = self._full_pipeline.transform(X_copy)
        return pd.DataFrame(X_prep, 
                            columns=self._columns, 
                            index=X_copy.index)

if __name__ == "__main__":
    df_train = load_nsl_kdd_csv(TRAIN_PATH)
    df_test = load_nsl_kdd_csv(TEST_PATH)

    if df_train is not None and df_test is not None:
        print("Tamaño de los datos de entrenamiento:", df_train.shape)
        print("Tamaño de los datos de prueba:", df_test.shape)
        
        print(df_train.head())
        print(df_train.dtypes)
        df = df_train.copy()
        df.info()

        print(df.describe())

        print(df["protocol_type"].value_counts())

        print(df["class"].value_counts())
        
        plt.figure()
        df["protocol_type"].value_counts().plot(kind="bar")
        plt.title("Distribución de protocol_type")
        plt.xlabel("protocol_type")
        plt.ylabel("Frecuencia")
        plt.savefig("distribucion_protocolo.png")

        df.hist(bins=50, figsize=(20,15))
        plt.savefig("histogramas.png")

        print("\n--- Transformación y Correlación ---")
        print("Aplicando Label Encoding a variables categóricas (solo para matriz de correlación)...")
        df_corr = df.copy()
        labelencoder = LabelEncoder()
        df_corr["class"] = labelencoder.fit_transform(df_corr["class"])
        df_corr["protocol_type"] = labelencoder.fit_transform(df_corr["protocol_type"])
        df_corr["service"] = labelencoder.fit_transform(df_corr["service"])
        df_corr["flag"] = labelencoder.fit_transform(df_corr["flag"])

        print("\nCorrelación con la variable 'class':")
        corr_matrix = df_corr.corr()
        print(corr_matrix["class"].sort_values(ascending=False))

        print("\nGenerando matriz de correlación...")
        fig, ax = plt.subplots(figsize=(12, 12))
        cax = ax.matshow(corr_matrix, cmap='coolwarm')
        fig.colorbar(cax)
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.savefig("matriz_correlacion.png", bbox_inches="tight")
        print("-> Guardado: matriz_correlacion.png")

        print("\nGenerando matriz de dispersión (Scatter Matrix)...")
        attributes = ["same_srv_rate", "dst_host_srv_count", "class", "dst_host_same_srv_rate"]
        scatter_matrix(df_corr[attributes], figsize=(12, 8))
        plt.savefig("scatter_matrix.png", bbox_inches="tight")
        print("-> Guardado: scatter_matrix.png")

        print("\nCorrelación final con 'class':")
        print(df_corr.corr()["class"].sort_values(ascending=False))

        print("\n--- División del Conjunto de Datos (Stratified Sampling) ---")
        print(f"Longitud del conjunto de datos: {len(df)}")

        # Usando la función de particionado completo
        train_set, val_set, test_set = train_val_test_split(df, stratify='protocol_type')

        print(f"Longitud del Training Set: {len(train_set)}")
        print(f"Longitud del Validation Set: {len(val_set)}")
        print(f"Longitud del Test Set: {len(test_set)}")
        
        print("\nComprobación de que stratify mantiene la proporción de la característica en los conjuntos...")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        df["protocol_type"].hist(ax=axes[0, 0])
        axes[0, 0].set_title("Original DataFrame")
        
        train_set["protocol_type"].hist(ax=axes[0, 1])
        axes[0, 1].set_title("Training Set")
        
        val_set["protocol_type"].hist(ax=axes[1, 0])
        axes[1, 0].set_title("Validation Set")
        
        test_set["protocol_type"].hist(ax=axes[1, 1])
        axes[1, 1].set_title("Test Set")
        
        plt.tight_layout()
        plt.savefig("stratified_histograms.png")
        print("-> Guardado: stratified_histograms.png")
        
        print("\n--- 3. Limpiando los datos ---")
        # Separamos las características de entrada de la característica de salida
        X_train = train_set.drop("class", axis=1)
        y_train = train_set["class"].copy()

        # Inyectamos valores nulos artificialmente para ilustrar el ejercicio
        X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"]<800), "src_bytes"] = np.nan
        X_train.loc[(X_train["dst_bytes"]>500) & (X_train["dst_bytes"]<2000), "dst_bytes"] = np.nan

        print("\nOpción 1: Eliminamos las filas con valores nulos")
        X_train_copy = X_train.copy()
        X_train_copy.dropna(subset=["src_bytes", "dst_bytes"], inplace=True)
        print("El número de filas eliminadas es:", len(X_train) - len(X_train_copy))

        print("\n--- 4. Transformación de atributos categóricos a numéricos ---")
        protocol_type = X_train_copy['protocol_type']
        protocol_type_encoded, categorias = protocol_type.factorize()
        print("Codificación de los primeros 10 elementos usando factorize():")
        for i in range(min(10, len(protocol_type_encoded))):
            print(f"{protocol_type.iloc[i]} = {protocol_type_encoded[i]}")
        print("Categorías detectadas:", categorias)

        print("\n--- One-Hot Encoding con Scikit-Learn ---")
        protocol_type_df = X_train_copy[['protocol_type']]
        oh_encoder = OneHotEncoder(handle_unknown='ignore')
        protocol_type_oh = oh_encoder.fit_transform(protocol_type_df)
        print(f"Dimensiones de la sparse matrix: {protocol_type_oh.shape}")
        print("Categorías detectadas (OneHotEncoder):", oh_encoder.categories_)

        print("\n--- One-Hot Encoding con Pandas (get_dummies) ---")
        print(pd.get_dummies(X_train_copy['protocol_type']).head())

        print("\n--- 5. Escalado del conjunto de datos ---")
        # Recuperamos el conjunto de datos limpio original (sin inyección de NaNs)
        X_train_clean = train_set.drop("class", axis=1)
        scale_attrs = X_train_clean[['src_bytes', 'dst_bytes']]

        robust_scaler = RobustScaler()
        X_train_scaled_array = robust_scaler.fit_transform(scale_attrs)
        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=['src_bytes', 'dst_bytes'])
        
        print("\nDatos escalados (head 10):")
        print(X_train_scaled.head(10))

        print("\n--- 6. Construyendo transformadores personalizados ---")
        # Usando el transformador de nulos
        delete_nan = DeleteNanRows()
        X_train_prep_nan = delete_nan.fit_transform(X_train)
        print(f"Dimensiones después de DeleteNanRows: {X_train_prep_nan.shape}")

        # Usando el escalador personalizado
        custom_scaler = CustomScaler(["src_bytes", "dst_bytes"])
        X_train_prep_custom = custom_scaler.fit_transform(X_train_prep_nan)
        print("Datos tras CustomScaler (head 5):")
        print(X_train_prep_custom[['src_bytes', 'dst_bytes']].head(5))

        print("\n--- 7. Construyendo Pipelines personalizados ---")
        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('rbst_scaler', RobustScaler()),
        ])
        num_attribs = list(X_train.select_dtypes(exclude=['object']))
        cat_attribs = list(X_train.select_dtypes(include=['object']))

        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
        ])
        X_train_prep_pipeline = full_pipeline.fit_transform(X_train)
        X_train_prep_pipeline_df = pd.DataFrame(X_train_prep_pipeline, columns=list(pd.get_dummies(X_train)), index=X_train.index)
        
        print("\nResultado final del Full Pipeline (head 10):")
        print(X_train_prep_pipeline_df.head(10))

        print("\n--- 8. Preparación Final del Conjunto de Datos ---")
        # Division del conjunto en los diferentes subconjuntos
        train_set, val_set, test_set = train_val_test_split(df)
        print("Longitud del Training Set:", len(train_set))
        print("Longitud del Validation Set:", len(val_set))
        print("Longitud del Test Set:", len(test_set))
        
        # Conjunto de datos general
        X_df = df.drop("class", axis=1)
        y_df = df["class"].copy()
        
        # Conjunto de datos de entrenamiento
        X_train = train_set.drop("class", axis=1)
        y_train = train_set["class"].copy()
        
        # Conjunto de datos de validación
        X_val = val_set.drop("class", axis=1)
        y_val = val_set["class"].copy()
        
        # Conjunto de datos de pruebas
        X_test = test_set.drop("class", axis=1)
        y_test = test_set["class"].copy()
        
        # Instanciamos nuestro transformador personalizado
        data_preparer = DataFramePreparer()
        
        # Hacemos el fit con el conjunto de datos general para que adquiera todos los valores posibles
        data_preparer.fit(X_df)
        
        # Transformamos el subconjunto de datos de entrenamiento
        X_train_prep = data_preparer.transform(X_train)
        print("\nX_train_prep (head 5):")
        print(X_train_prep.head(5))
        print("Dimensiones de X_train_prep:", X_train_prep.shape)
        
        # Transformamos el subconjunto de datos de validacion
        X_val_prep = data_preparer.transform(X_val)
        print("\nX_val_prep (head 5):")
        print(X_val_prep.head(5))
        print("Dimensiones de X_val_prep:", X_val_prep.shape)
        
        print("\n--- 9. Entrenamiento del Modelo (Regresión Logística) ---")
        # Entrenamos un algoritmo basado en regresión logística
        clf = LogisticRegression(solver="newton-cg", max_iter=1000)
        clf.fit(X_train_prep, y_train)
        print("Modelo entrenado exitosamente.")

        print("\n--- 10. Predicción y Evaluación (Validation Set) ---")
        y_pred_val = clf.predict(X_val_prep)
        
        print("Matriz de Confusión (Validation):")
        print(confusion_matrix(y_val, y_pred_val))

        # Como revertimos el LabelEncoder en df, podemos volver a usar la etiqueta original de texto
        pos_label_anomaly = 'anomaly'
        
        print("Precisión:", precision_score(y_val, y_pred_val, pos_label=pos_label_anomaly))
        print("Recall:", recall_score(y_val, y_pred_val, pos_label=pos_label_anomaly))
        print("F1 score:", f1_score(y_val, y_pred_val, pos_label=pos_label_anomaly))
        
        # Guardado de gráficos (Validation)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(clf, X_val_prep, y_val, values_format='d', ax=ax)
        plt.title("Matriz de Confusión - Validation")
        plt.savefig("confusion_matrix_val.png", bbox_inches="tight")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(clf, X_val_prep, y_val, ax=ax)
        plt.title("Curva ROC - Validation")
        plt.savefig("roc_curve_val.png", bbox_inches="tight")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay.from_estimator(clf, X_val_prep, y_val, ax=ax)
        plt.title("Curva PR - Validation")
        plt.savefig("pr_curve_val.png", bbox_inches="tight")
        
        print("-> Gráficos de evaluación de validación guardados.")

        print("\n--- 11. Evaluación del modelo con el conjunto de datos de pruebas ---")
        X_test_prep = data_preparer.transform(X_test)
        y_pred_test = clf.predict(X_test_prep)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(clf, X_test_prep, y_test, values_format='d', ax=ax)
        plt.title("Matriz de Confusión - Test")
        plt.savefig("confusion_matrix_test.png", bbox_inches="tight")
        
        print("F1 score (Test):", f1_score(y_test, y_pred_test, pos_label=pos_label_anomaly))
        