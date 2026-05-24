"""
Demo: lectura de archivos CSV de la carpeta FCPS, extracción de meta-features y normalización min-max.

Procesa cada archivo CSV en data/datasets/FCPS.
La primera columna es la clase (y), el resto son atributos (X).
"""

import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cash.SelectorModeloClustering import SelectorModeloClustering
from src.cash.SelectorModeloClasificacion import SelectorModeloClasificacion
from src.cash.SelectorModeloRegresion import SelectorModeloRegresion

# Añadir raíz del proyecto al path para imports absolutos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q_table import QTable
from pipeline_q_table import PipelineQTable
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

def cargar_dataset_fcps(ruta_csv: str) -> tuple[str, str, pd.DataFrame, pd.Series]:
    """
    Carga un dataset CSV de la carpeta FCPS.
    La primera columna es la clase (y), el resto son atributos (X).
    
    Retorna: (nombre_dataset, descripcion, X_df, y_series)
    """
    df = pd.read_csv(ruta_csv)
    
    # Primera columna es y (clase)
    y_series = df.iloc[:, 0]
    
    # El resto son X (atributos)
    X_df = df.iloc[:, 1:]
    
    # Nombre del dataset es el nombre del archivo sin extensión
    nombre_dataset = os.path.splitext(os.path.basename(ruta_csv))[0]
    
    # Descripción es una cadena vacía
    descripcion = ""
    
    return nombre_dataset, descripcion, X_df, y_series

# -------------------------------------------------------------------
# Procesamiento principal
# -------------------------------------------------------------------
def main():
    extractor = ExtractorMetaFeatures()
    extractor.silenciar_warnings_pymfe()

    # Cargar parámetros de normalización
    ruta_params = os.path.join("data", "normalizacion", "normalization_params.csv")
    if not os.path.exists(ruta_params):
        print(f"[ERROR] No se encuentra {ruta_params}")
        return
    params_norm = cargar_params_normalizacion(ruta_params)
    print(f"[INFO] Parámetros de normalización cargados: {len(params_norm)} dimensiones")

    # Directorio de datasets FCPS
    directorio_datasets = os.path.join("data", "FCPS")
    if not os.path.exists(directorio_datasets):
        print(f"[ERROR] No se encuentra la carpeta {directorio_datasets}")
        return
    
    archivos_csv = [f for f in os.listdir(directorio_datasets) if f.endswith(".csv")]
    archivos_csv.sort()
    
    if not archivos_csv:
        print(f"[ERROR] No se encontraron archivos .csv en {directorio_datasets}")
        return
    
    print(f"[INFO] Se encontraron {len(archivos_csv)} archivos CSV: {archivos_csv}")

    for archivo in archivos_csv:
        ruta_archivo = os.path.join(directorio_datasets, archivo)
        print("\n" + "=" * 70)
        print(f"[INFO] Procesando archivo: {archivo}")
        print("=" * 70)

        tipo_tarea = "clustering"
        print(f"[INFO] Tipo de tarea: {tipo_tarea}")

        tiempo_inicio = time.perf_counter()
        print(f"\n  --- Archivo: {archivo} ---")

        # 1. Cargar dataset desde CSV
        try:
            nombre_dataset, descripcion, X_df, y_series = cargar_dataset_fcps(ruta_archivo)
            print(f"  Dataset: {nombre_dataset}  |  #filas: {X_df.shape[0]}  |  #cols: {X_df.shape[1]}")
        except Exception as e:
            print(f"  [ERROR] Carga del dataset fallida: {e}")
            continue

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
        existen_acciones_clustering = False
        contador = 1
        while not existen_acciones_clustering:
            q_table = QTable()
            cluster = q_table.determinar_cluster(vector_normalizado, contador)
            print(f"  Cluster asignado: {cluster}")

            # 5. Crear guía de estados y acciones recomendadas para cada fase
            guia_estado_acciones = {}
            for i in range(16):
                estado_actual = f"{cluster}_fase{i}"
                acciones_recomendadas = q_table.obtener_acciones_ordenadas(estado_actual).copy()
                guia_estado_acciones[estado_actual] = acciones_recomendadas
                print(f"    Estado: {estado_actual}  |  Acciones recomendadas: {acciones_recomendadas[:3]}")  # Mostrar solo top 3 acciones

            if len(guia_estado_acciones) < 16:
                print(f"  [ERROR] No se encontraron suficientes estados en la Q-table para el cluster {cluster}. Se esperaban 16, pero se encontraron {len(guia_estado_acciones)}.")
                contador += 1
                continue
                
            existen_acciones_clustering = True

        # 6. Dividir en 80/20 para entrenamiento/prueba
        X_procesado = X_df.copy()
        y_procesado = y_series.copy()

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
                X_temporal, y_temporal = X_procesado.copy(), y_procesado.copy()

                accion = acciones_recomendadas.pop(0)
                
                accion_limpia = accion.replace(f"{fase_actual}_", "")
                instancia = fases_instancias[fase_actual]
                instancia.log_algoritmo = accion_limpia if accion_limpia != 'None' else None

                try:
                    if fase_actual == "crear_nueva_variable" and descripcion:
                        instancia.descripcion = descripcion
                    elif fase_actual == "seleccionar_variables" and descripcion:
                        instancia.descripcion = descripcion

                    instancia.fit(X_temporal, y_temporal)
                    X_temporal, y_temporal = instancia.transform(X_temporal, y_temporal)

                    print(f"[INFO] Fase: {fase_actual:<45} | Acción aplicada: {accion}")

                    accion_exitosa = True
                except Exception as e:
                    print(f"[ERROR] Fase: {fase_actual:<45} | Acción '{accion}' fallida: {e}")
                    continue
            
            if not accion_exitosa:
                print("PIPELINE ROTO: No se pudo aplicar ninguna acción recomendada para esta fase.")
                break
            
            X_procesado, y_procesado = X_temporal, y_temporal
        
        if not accion_exitosa:
            print("[ERROR] No se pudo completar el pipeline para este dataset debido a errores en las fases.")
            registrador.guardar_resultado(
                nombre_automl="q_learning_v1",
                task_id=None,
                nombre_dataset=nombre_dataset,
                fuente="FCPS",
                cluster_nombre=cluster,
                tiempo=time.perf_counter() - tiempo_inicio,
                metricas={},  # será ignorado
                formula_recompensa="original-clustering",
                dataset_imposible=True,  # <-- activa el -1111
            )
            continue
        # ===============================================
        # Antecedente: Calcular MTFs globales para la selección de modelo
        # ===============================================
        extractor = ExtractorMetaFeatures()
        meta_features_globales, _ = extractor.extraer_desde_dataframe(X_procesado.copy(), y_procesado.copy(), vectorizar=False)
        meta_features_globales = extractor.eliminar_constantes_errores(meta_features_globales)
        meta_features_globales_formateadas = extractor.formatear_meta_features_globales(meta_features_globales)

        # 8. Seleccionar LLM y calculo de hiperparámetros
        accion_exitosa = False
        acciones_recomendadas = guia_estado_acciones[estados[12]].copy()

        while not accion_exitosa and len(acciones_recomendadas) > 0:
            llm_recomendado = acciones_recomendadas.pop(0)
            llm_recomendado = llm_recomendado if llm_recomendado != 'ninguno' else None

            print(f"[INFO] LLM recomendado: {llm_recomendado}")

            modelos_recomendados = guia_estado_acciones[estados[15]].copy()
            selector = SelectorModeloClustering()

            modelo_exitoso = False
            while not modelo_exitoso and len(modelos_recomendados) > 0:
                modelo_recomendado = modelos_recomendados.pop(0)
                modelo_recomendado = modelo_recomendado.replace("clustering_", "").strip()
                
                try:
                    selector.log_algoritmo = modelo_recomendado
                    selector.llm_seleccionado = llm_recomendado

                    selector.calcular_hiper_parametros(X_procesado.copy(), meta_features_globales_formateadas)
                    modelo_result = selector.entrenar_modelo(X_procesado.copy())
                    
                    if modelo_result.is_failure:
                        print(f"[ERROR] Modelo '{modelo_recomendado}' falló al entrenar: {modelo_result.get_error()}")
                        continue

                except Exception as e:
                    print(f"[ERROR] Modelo '{modelo_recomendado}' falló al entrenar/evaluar: {e}")
                    continue

                modelo = modelo_result.get_value()

                # 9. Evaluar modelo
                evaluador = EvaluadorModelos()
                metricas = evaluador.evaluar_un_modelo_clustering(modelo, X_procesado.copy(), y_procesado.copy())
                
                if metricas is None:
                    print(f"[ERROR] Modelo '{modelo_recomendado}' falló al evaluar (métricas con -1111): {metricas}")
                    continue

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
                task_id=None,
                nombre_dataset=nombre_dataset,
                fuente="FCPS",
                cluster_nombre=cluster,
                tiempo=time.perf_counter() - tiempo_inicio,
                metricas={},  # será ignorado
                formula_recompensa="original-clustering",
                dataset_imposible=True,  # <-- activa el -1111
            )
            continue
        
        print(f"[INFO] Dataset '{nombre_dataset}' procesado exitosamente con LLM '{llm_recomendado}' y modelo '{modelo_recomendado}'.")
        print(f"Métricas finales:\n{metricas}")
        registrador.guardar_resultado(
            nombre_automl="q_learning_v1",
            task_id=None,
            nombre_dataset=nombre_dataset,
            fuente="FCPS",
            cluster_nombre=cluster,
            tiempo=time.perf_counter() - tiempo_inicio,
            metricas=metricas,
            formula_recompensa="original-clustering",
            dataset_imposible=False,
        )
        

    print("\n[INFO] Demo completada.")

if __name__ == "__main__":
    main()