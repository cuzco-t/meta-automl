"""
Demo: lectura de task_ids, descarga de datasets, extracción de meta-features y normalización min-max.

Procesa únicamente los primeros 5 task_id de cada archivo .txt en data/datasets.
"""

import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cash.SelectorModeloClasificacion import SelectorModeloClasificacion
from src.cash.SelectorModeloRegresion import SelectorModeloRegresion

# Añadir raíz del proyecto al path para imports absolutos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q_table import QTable
from pipeline_q_table import PipelineQTable
from src.openml_descargador import OpenMLDescargador
from src.ExtractorMetaFeatures import ExtractorMetaFeatures

from src.minero.evaluador_modelos import EvaluadorModelos
from src.preprocesamiento.TratarDuplicados import TratarDuplicados
from src.preprocesamiento.CodificarVariablesBinarias import CodificarVariablesBinarias
from src.preprocesamiento.TratarFaltantesNumericos import TratarFaltantesNumericos
from src.preprocesamiento.TratarFaltantesStrings import TratarFaltantesStrings
from src.preprocesamiento.CodificarVariablesCategoricasRangoBajo import CodificarVariablesCategoricasRangoBajo
from src.preprocesamiento.CodificarVariablesCategoricasRangoMedio import CodificarVariablesCategoricasRangoMedio
from src.preprocesamiento.CodificarVariablesCategoricasRangoAlto import CodificarVariablesCategoricasRangoAlto
from src.preprocesamiento.TratarOutliersNumericos import TratarOutliersNumericos
from src.preprocesamiento.EscalarDatosNumericos import EscalarDatosNumericos
from src.preprocesamiento.NormalizarDatosNumericos import NormalizarDatosNumericos
from src.preprocesamiento.CrearNuevaVariable import CrearNuevaVariable
from src.preprocesamiento.SeleccionarVariables import SeleccionarVariables

from src.RegistradorQTable import RegistradorQTable
registrador = RegistradorQTable()

def crear_fases_instancias() -> dict[str, object]:
    """Retorna nuevas instancias de cada fase."""
    return {
        "tratar_duplicados": TratarDuplicados(),
        "codificar_variables_binarias": CodificarVariablesBinarias(),
        "tratar_faltantes_numericos": TratarFaltantesNumericos(),
        "tratar_faltantes_strings": TratarFaltantesStrings(),
        "codificar_variables_categoricas_rango_bajo": CodificarVariablesCategoricasRangoBajo(),
        "codificar_variables_categoricas_rango_medio": CodificarVariablesCategoricasRangoMedio(),
        "codificar_variables_categoricas_rango_alto": CodificarVariablesCategoricasRangoAlto(),
        "tratar_outliers_numericos": TratarOutliersNumericos(),
        "escalar_datos_numericos": EscalarDatosNumericos(),
        "normalizar_datos_numericos": NormalizarDatosNumericos(),
        "crear_nueva_variable": CrearNuevaVariable(),
        "seleccionar_variables": SeleccionarVariables()
    }
# -------------------------------------------------------------------
# Constantes de error
# -------------------------------------------------------------------
BANDERA_ERROR = -1111.0
BANDERA_INFINITO = 2222.0

# -------------------------------------------------------------------
# Función para cargar parámetros de normalización
# -------------------------------------------------------------------
def cargar_params_normalizacion(ruta_csv: str) -> dict[int, tuple[float, float]]:
    """
    Retorna un dict:  {indice_dimension: (min, max)}
    """
    df = pd.read_csv(ruta_csv)
    params = {}
    for _, row in df.iterrows():
        dim = int(row["dimension"])
        min_val = float(row["min"])
        max_val = float(row["max"])
        params[dim] = (min_val, max_val)
    return params

# -------------------------------------------------------------------
# Función de normalización min-max respetando banderas de error/infinito
# -------------------------------------------------------------------
def normalizar_vector(vector: list[float],
                      params: dict[int, tuple[float, float]]) -> list[float]:
    """
    Normaliza cada elemento del vector usando los parámetros min/max de su dimensión.

    Reglas:
      - Si el valor es -1111.0 (bandera de error) o 2222.0 (infinito),
        se deja sin normalizar (para reemplazar después).
      - Si no, se aplica min-max:  (v - min) / (max - min)
      - Finalmente, se reemplazan -1111 -> -1  y  2222 -> 2
    """
    resultado = []
    for i, valor in enumerate(vector):
        if i not in params:
            # Si la dimensión no está en los parámetros, se deja igual
            resultado.append(valor)
            continue

        v_min, v_max = params[i]

        # Si es bandera de error/infinito -> no normalizar
        if valor == BANDERA_ERROR or valor == BANDERA_INFINITO:
            resultado.append(valor)
        else:
            # Normalización min-max
            if v_max - v_min == 0:
                norm = 0.0
            else:
                norm = (valor - v_min) / (v_max - v_min)
            resultado.append(norm)

    # Segundo paso: reemplazar banderas
    resultado = [-1.0 if x == BANDERA_ERROR else x for x in resultado]
    resultado = [2.0 if x == BANDERA_INFINITO else x for x in resultado]

    return resultado

def agregar_dimensiones_tarea(vector: list[float], tipo_tarea: str) -> list[float]:
    """
    Agrega dimensiones adicionales al vector de meta-features para indicar el tipo de tarea.

    Por ejemplo:
      - Para clasificación: [1, 0, 0]
      - Para regresión: [0, 1, 0]
      - Para clustering: [0, 0, 1]

    Estas dimensiones pueden ayudar a los modelos a diferenciar entre tipos de tareas.
    """
    if tipo_tarea == "clasificación":
        return vector + [1.0, 0.0, 0.0]
    elif tipo_tarea == "regresión":
        return vector + [0.0, 1.0, 0.0]
    elif tipo_tarea == "clustering":
        return vector + [0.0, 0.0, 1.0]

# -------------------------------------------------------------------
# Procesamiento principal
# -------------------------------------------------------------------
def main():
    descargador = OpenMLDescargador()
    extractor = ExtractorMetaFeatures()
    extractor.silenciar_warnings_pymfe()

    # Cargar parámetros de normalización
    ruta_params = os.path.join("data", "normalizacion", "normalization_params.csv")
    if not os.path.exists(ruta_params):
        print(f"[ERROR] No se encuentra {ruta_params}")
        return
    params_norm = cargar_params_normalizacion(ruta_params)
    print(f"[INFO] Parámetros de normalización cargados: {len(params_norm)} dimensiones")

    # Directorio de datasets
    directorio_datasets = os.path.join("data", "datasets")
    archivos_txt = [f for f in os.listdir(directorio_datasets) if f.endswith(".txt")]
    archivos_txt.sort()

    for archivo in archivos_txt:
        ruta_archivo = os.path.join(directorio_datasets, archivo)
        print("\n" + "=" * 70)
        print(f"[INFO] Procesando archivo: {archivo}")
        print("=" * 70)

        # Determinar tipo de tarea
        if "cc18" in archivo:
            tipo_tarea = "clasificación"
        elif "ctr23" in archivo:
            tipo_tarea = "regresión"
        else:
            tipo_tarea = "desconocido"
        print(f"[INFO] Tipo de tareas: {tipo_tarea}")

        # Leer task_ids
        with open(ruta_archivo, "r") as f:
            task_ids = [int(line.strip()) for line in f if line.strip()]

        # Solo primeros 5
        task_ids = task_ids[:]
        print(f"[INFO] Task IDs a procesar: {task_ids}")

        for tid in task_ids:
            tiempo_inicio = time.perf_counter()
            print(f"\n  --- Task ID: {tid} ---")

            # 1. Descargar dataset
            resultado = descargador.obtener_datos_tarea(tid)
            if resultado.is_failure:
                print(f"  [ERROR] Descarga fallida: {resultado.get_error()}")
                continue

            nombre_dataset, descripcion, X_df, y_series = resultado.get_value()
            print(f"  Dataset: {nombre_dataset}  |  #filas: {X_df.shape[0]}  |  #cols: {X_df.shape[1]}")

            # 2. Extraer meta-features (vectorizadas)
            try:
                _, vector = extractor.extraer_desde_dataframe(X_df, y_series, vectorizar=True)
            except Exception as e:
                print(f"  [ERROR] Extracción de meta-features fallida: {e}")
                continue

            if vector is None:
                print("  [ERROR] El vector de meta-features es None")
                continue
            
            vector = agregar_dimensiones_tarea(vector, tipo_tarea)
            print(f"  Longitud del vector original: {len(vector)}")

            # 3. Normalizar
            vector_normalizado = normalizar_vector(vector, params_norm)
            print(f"  Vector normalizado (primeros 5 elementos): {vector_normalizado[:5]}")

            # 4. Determinar cluster
            q_table = QTable()
            cluster = q_table.determinar_cluster(vector_normalizado)
            print(f"  Cluster asignado: {cluster}")

            # 5. Crear guía de estados y acciones recomendadas para cada fase
            guia_estado_acciones = {}
            for i in range(16):
                estado_actual = f"{cluster}_fase{i}"
                acciones_recomendadas = q_table.obtener_acciones_ordenadas(estado_actual).copy()
                guia_estado_acciones[estado_actual] = acciones_recomendadas
                print(f"    Estado: {estado_actual}  |  Acciones recomendadas: {acciones_recomendadas[:3]}")  # Mostrar solo top 3 acciones

            # 6. Dividir en 80/20 para entrenamiento/prueba
            X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=912)

            # 7. Preprocesamiento
            estados = list(guia_estado_acciones.keys())
            
            fases_instancias = crear_fases_instancias()
            fases_str = list(fases_instancias.keys())
            for i in range(12):
                estado_actual = estados[i]
                fase_actual = fases_str[i]
                acciones_recomendadas = guia_estado_acciones[estado_actual].copy()


                accion_exitosa = False
                while not accion_exitosa and len(acciones_recomendadas) > 0:
                    X_train_temp, y_train_temp = X_train.copy(), y_train.copy()
                    X_val_temp, y_val_temp = X_test.copy(), y_test.copy()

                    accion = acciones_recomendadas.pop(0)
                    
                    accion_limpia = accion.replace(f"{fase_actual}_", "")
                    instancia = fases_instancias[fase_actual]
                    instancia.log_algoritmo = accion_limpia if accion_limpia != 'None' else None

                    try:
                        if fase_actual == "crear_nueva_variable" and descripcion:
                            instancia.descripcion = descripcion
                        elif fase_actual == "seleccionar_variables" and descripcion:
                            instancia.descripcion = descripcion

                        instancia.fit(X_train_temp, y_train_temp)
                        X_train_temp, y_train_temp = instancia.transform(X_train_temp, y_train_temp)
                        X_val_temp, y_val_temp = instancia.transform(X_val_temp, y_val_temp)

                        print(f"[INFO] Fase: {fase_actual:<45} | Acción aplicada: {accion}")

                        accion_exitosa = True
                    except Exception as e:
                        print(f"[ERROR] Fase: {fase_actual:<45} | Acción '{accion}' fallida: {e}")
                        continue
                
                if not accion_exitosa:
                    print("PIPELINE ROTO: No se pudo aplicar ninguna acción recomendada para esta fase.")
                    break
                
                X_train, y_train = X_train_temp, y_train_temp
                X_test, y_test = X_val_temp, y_val_temp
            
            if not accion_exitosa:
                print("[ERROR] No se pudo completar el pipeline para este dataset debido a errores en las fases.")
                registrador.guardar_resultado(
                    nombre_automl="q_learning_v1",
                    task_id=tid,
                    nombre_dataset=nombre_dataset,
                    fuente=archivo,
                    cluster_nombre=cluster,
                    tiempo=time.perf_counter() - tiempo_inicio,
                    metricas={},  # será ignorado
                    formula_recompensa="original",
                    dataset_imposible=True,  # <-- activa el -1111
                )
                continue
            # ===============================================
            # Antecedente: Calcular MTFs globales para la selección de modelo
            # ===============================================
            X_total = pd.concat([X_train, X_test], axis=0, ignore_index=True)
            y_total = pd.concat([y_train, y_test], axis=0, ignore_index=True)

            extractor = ExtractorMetaFeatures()
            meta_features_globales, _ = extractor.extraer_desde_dataframe(X_total, y_total)
            meta_features_globales = extractor.eliminar_constantes_errores(meta_features_globales)
            meta_features_globales_formateadas = extractor.formatear_meta_features_globales(meta_features_globales)

            # 8. Seleccionar LLM y calculo de hiperparámetros
            accion_exitosa = False
            acciones_recomendadas = guia_estado_acciones[estados[12]].copy()

            while not accion_exitosa and len(acciones_recomendadas) > 0:
                llm_recomendado = acciones_recomendadas.pop(0)
                llm_recomendado = llm_recomendado if llm_recomendado != 'ninguno' else None

                print(f"[INFO] LLM recomendado: {llm_recomendado}")

                modelos_recomendados = []
                selector = None
                if tipo_tarea == "clasificación":
                    modelos_recomendados = guia_estado_acciones[estados[13]].copy()
                    selector = SelectorModeloClasificacion()
                elif tipo_tarea == "regresión":
                    modelos_recomendados = guia_estado_acciones[estados[14]].copy()
                    selector = SelectorModeloRegresion()

                modelo_exitoso = False
                while not modelo_exitoso and len(modelos_recomendados) > 0:
                    modelo_recomendado = modelos_recomendados.pop(0)
                    modelo_recomendado = modelo_recomendado.replace("clasificacion_", "").replace("regresion_", "").strip()
                    
                    try:
                        selector.log_algoritmo = modelo_recomendado
                        selector.llm_seleccionado = llm_recomendado

                        selector.calcular_hiper_parametros(X_total, y_total, meta_features_globales_formateadas)
                        modelo_result = selector.entrenar_modelo(X_train, y_train)
                        
                        if modelo_result.is_failure:
                            print(f"[ERROR] Modelo '{modelo_recomendado}' falló al entrenar: {modelo_result.get_error()}")
                            continue

                    except Exception as e:
                        print(f"[ERROR] Modelo '{modelo_recomendado}' falló al entrenar/evaluar: {e}")
                        continue

                    modelo = modelo_result.get_value()

                    # 9. Evaluar modelo
                    evaluador = EvaluadorModelos()
                    metricas = evaluador.evaluar_un_modelo_supervisado(modelo, X_test, y_test, tipo_tarea)
                    print(f"  Modelo: {modelo_recomendado}  |  Métricas: {metricas}")
                    modelo_exitoso = True
                
                if not modelo_exitoso:
                    print(f"[ERROR] No se pudo entrenar/evaluar ningún modelo con el llm {llm_recomendado}.")
                    continue

                accion_exitosa = True
            
            if not accion_exitosa:
                print("DATASET IMPOSIBLE DE PROCESAR")
                registrador.guardar_resultado(
                    nombre_automl="q_learning_v1",
                    task_id=tid,
                    nombre_dataset=nombre_dataset,
                    fuente=archivo,
                    cluster_nombre=cluster,
                    tiempo=time.perf_counter() - tiempo_inicio,
                    metricas={},  # será ignorado
                    formula_recompensa="original",
                    dataset_imposible=True,  # <-- activa el -1111
                )
                continue
            
            print(f"[INFO] Task ID {tid} procesada exitosamente con LLM '{llm_recomendado}' y modelo '{modelo_recomendado}'.")
            print(f"Métricas finales:\n{metricas}")
            registrador.guardar_resultado(
                nombre_automl="q_learning_v1",
                task_id=tid,
                nombre_dataset=nombre_dataset,
                fuente=archivo,
                cluster_nombre=cluster,
                tiempo=time.perf_counter() - tiempo_inicio,
                metricas=metricas,
                formula_recompensa="original",
                dataset_imposible=False,
            )
            
        print(f"\n[INFO] Datasets de tipo '{tipo_tarea}' procesados.")
        

    print("\n[INFO] Demo completada.")

if __name__ == "__main__":
    main()