import ollama
import pandas as pd
import requests.exceptions

from .config.Configuracion import Configuracion

from toon_format import encode

class LLM:
    def __init__(self):
        # timeout en segundos para la llamada al servidor
        config = Configuracion()

        self.llm_host = config.llm_host
        self.llm_timeout = config.llm_timeout

        self.client = ollama.Client(
            host=config.llm_host,
            timeout=config.llm_timeout
        )
        self.modelo = config.llm_modelo
        self.num_ctx = config.llm_num_ctx
    
    def generar_respuesta(self, prompt):
        try:
            respuesta = self.client.chat(
                model=self.modelo,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": self.num_ctx  # Ajusta el contexto del modelo si es necesario
                }
            )
            
            return respuesta.message.content

        except requests.exceptions.Timeout:
            raise TimeoutError(f"El modelo no respondió en {self.client.timeout} segundos")

        except Exception as e:
            # aquí puedes capturar cualquier otro fallo
            raise RuntimeError(f"Error al generar respuesta: {e}")
    
    def plantillas_prompts(self, plantilla, **kwargs):

        def seleccionar_variables():
            prompt = f"""
Eres un experto en selección de variables para modelos de ML en tareas de {kwargs["kwargs"]["tarea"]}. 
Analiza estas meta‑features (formato TOON) por columna que extraje con pymfe:
{kwargs["kwargs"]["meta_features_por_columna"]}

Devuélveme **solo una lista Python con los nombres de las variables** que recomiendas para entrenar el modelo.
Ejemplo: ["variable1", "variable2", "variable3"]
Sin texto adicional ni explicación.
"""

            return prompt
        
        def seleccionar_hiper_parametros():
            prompt = f"""
Actua como un experto en modelos de ML de la libreria scikit-learn, quiero que analices las siguientes meta-features,
que consegui utilizando la libreria pymfe. Los las meta-features globales del datset. Con base en ellas, y considerando
que estoy relizando una tarea de {kwargs["kwargs"]["tarea"]}, con un modelo {kwargs["kwargs"]["modelo_ml"]}, 
quiero que me ayudes recomendando valores para los hiperparámetros del modelo. A continuacion de paso las 
meta-features globales, y la configuracion de hiperparámetros por defecto para que me ayudes a reemplazar sus valores.

meta-features globales: {kwargs["kwargs"]["meta_features_globales"]}

condifugracion de hiperparametros por defecto: {kwargs["kwargs"]["hiper_parametros_por_defecto"]}

IMPORTANTE: Toma en cuenta que tu respuesta debe ser unicamente un diccionario python con los nombres
de los hiperparámetros y sus valores recomendados, sin ningun tipo de texto adicional, ni explicacion,
ni formato diferente a un diccionario python.
Aqui tienes un ejemplo de como debe ser tu respuesta: {{'hiperparametro_1': valor_1, 'hiperparametro_2': valor_2}}
"""
            return prompt
        
        def crear_nueva_variable():
            prompt = f"""
Actúa como un experto en feature engineering para machine learning. Estoy trabajando con un dataset
de {kwargs["tarea"]} y necesito generar una nueva variable que pueda mejorar el rendimiento de un modelo. 

Te proporcionaré:
1. La lista de columnas del dataset.
2. Una descripción del dataset (puede estar vacía; si es así, usa únicamente los nombres de las columnas
para inferir relaciones).

Tu tarea es generar hasta 5 nuevas variables siguiendo estas reglas:

- Devuelve **únicamente** un diccionario de Python sin ningún bloque de código (python), explicación,
texto adicional, ni comentarios. Las reglas del diccionario son:
    - La clave es el nombre de la nueva variable.
    - El valor es un string con la operación aritmética que define la nueva variable usando
    los nombres exactos de las columnas proporcionadas encerradas entre backticks. 
    Esta operación debe ser válida como entrada para eval() en pandas.
- No es obligatorio generar 5 variables; pueden ser entre 1 y 5, dependiendo de lo que
tenga sentido según la información disponible.
- Las operaciones deben tener sentido en función de la descripción del dataset y/o 
los nombres de las columnas, buscando crear características útiles para {kwargs["tarea"]}.
- Evita generar variables que sean trivialmente iguales a columnas existentes o combinaciones redundantes.
- Si no encuentras ninguna combinación o transformación útil, devuelve un diccionario vacío {{}}.

Ejemplo:
Si las columnas fueran ["precio_unitario", "cantidad"], una posible respuesta podría ser:
{{"total": "(`precio_unitario` * `cantidad`)"}}

Ahora, usando la información de mi dataset, genera las nuevas variables.

Columnas: {kwargs["columnas"]}
Descripción: {kwargs["descripcion"]}
"""
            return prompt

        diccionario_plantilla_prompt = {
            "seleccionar_variables": seleccionar_variables,
            "seleccionar_hiper_parametros": seleccionar_hiper_parametros,
            "crear_nueva_variable": crear_nueva_variable
        }

        return diccionario_plantilla_prompt[plantilla]()

