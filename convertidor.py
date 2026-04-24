import pandas as pd

def arff_to_csv_fast(archivo_arff, archivo_csv):
    print(f"Procesando: {archivo_arff}")
    datos = []
    en_seccion_data = False
    
    with open(archivo_arff, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            if linea.lower().startswith('@data'):
                en_seccion_data = True
                continue
            
            if en_seccion_data:
                # El ARFF separa por comas. Limpiamos espacios y quitamos puntos finales si existen
                fila = [valor.strip().strip('.') for valor in linea.split(',')]
                datos.append(fila)
    
    # Lista completa de las columnas (43 nombres)
    columnas_completas = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"
    ]
    
   
    num_columnas_reales = len(datos[0])


    columnas_finales = columnas_completas[:num_columnas_reales]
    

    df = pd.DataFrame(datos, columns=columnas_finales)
    df.to_csv(archivo_csv, index=False)

arff_to_csv_fast('KDDTrain+.arff', 'KDDTrain+.csv')
arff_to_csv_fast('KDDTest+.arff', 'KDDTest+.csv')