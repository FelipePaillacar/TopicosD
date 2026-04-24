from google.colab import drive
import os
import pandas as pd

drive.mount("/content/drive")

DATA_DIR = "/content/drive/MyDrive/Datasets/NSL-KDD"
TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")

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

# 4. Verificación de existencia
print("Ruta de entrenamiento:", TRAIN_PATH)
print("Ruta de prueba:", TEST_PATH)
print("¿Existe TRAIN_PATH?", os.path.exists(TRAIN_PATH))
print("¿Existe TEST_PATH?", os.path.exists(TEST_PATH))
print("Número de columnas esperadas:", len(COLUMN_NAMES))

# 5. Función para cargar con Pandas
def load_nsl_kdd_txt(data_path):
    return pd.read_csv(data_path, header=None, names=COLUMN_NAMES)

# 6. Lectura nativa cruda (como se solicitaba en la explicación)
if os.path.exists(TRAIN_PATH):
    with open(TRAIN_PATH, "r") as train_set:
        lineas = train_set.readlines()
    print(f"\nNúmero de líneas leídas: {len(lineas)}")
    print("Primeras 3 líneas:")
    print(lineas[:3])
