import os
import pandas as pd

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "class", "difficulty"
]

def cargar_dataset(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        print(f" Error: No se encontró el archivo '{ruta_archivo}'. Asegúrate de que esté en la misma carpeta.")
        return None
    
    print(f"Cargando {ruta_archivo}...")
    # Pandas lee el txt directamente asumiendo que está separado por comas
    df = pd.read_csv(ruta_archivo, header=None, names=COLUMN_NAMES)
    print(f"¡Cargado exitosamente! Dimensiones: {df.shape}")
    return df

if __name__ == "__main__":
    ARCHIVO_TRAIN = "KDDTrain+.txt"
    ARCHIVO_TEST = "KDDTest+.txt"

    df_train = cargar_dataset(ARCHIVO_TRAIN)
    df_test = cargar_dataset(ARCHIVO_TEST)

    if df_train is not None:
        print("\n--- Vista previa de KDDTrain+ ---")
        print(df_train.head())
        print("\nValores únicos en la columna 'class':")
        print(df_train['class'].value_counts().head())
