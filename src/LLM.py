import ollama
import pandas as pd
import requests.exceptions

from toon_format import encode

class LLM:
    def __init__(self, modelo, timeout=10_800):
        # timeout en segundos para la llamada al servidor

        self.client = ollama.Client(timeout=timeout)
        self.modelo = modelo
    
    def generar_respuesta(self, prompt):
        try:
            respuesta = self.client.chat(
                model=self.modelo,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 20_000  # Ajusta el contexto del modelo si es necesario
                }
            )
            print("Respuesta del modelo:")
            print(respuesta.message.content)
            
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
Columnas:
{kwargs["kwargs"]["meta_features_por_columna"]["columnas"]}
Meta-features por columna:
{kwargs["kwargs"]["meta_features_por_columna"]["valores_meta_features"]}

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

        diccionario_plantilla_prompt = {
            "seleccionar_variables": seleccionar_variables,
            "seleccionar_hiper_parametros": seleccionar_hiper_parametros
        }

        return diccionario_plantilla_prompt[plantilla]()

