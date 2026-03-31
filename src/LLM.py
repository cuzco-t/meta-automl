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
            
            
            if "```python" in respuesta.message.content:
                # Extraer el código del bloque
                start = respuesta.message.content.find("```python") + len("```python")
                end = respuesta.message.content.find("```", start)
                respuesta.message.content = respuesta.message.content[start:end].strip()

            print(f"Respuesta del modelo: {respuesta.message.content}")  # Para depuración
            return respuesta.message.content

        except requests.exceptions.Timeout:
            raise TimeoutError(f"El modelo no respondió en {self.client.timeout} segundos")

        except Exception as e:
            # aquí puedes capturar cualquier otro fallo
            raise RuntimeError(f"Error al generar respuesta: {e}")
    
    def plantillas_prompts(self, plantilla, **kwargs):

        def seleccionar_variables():
            prompt = f"""
You are an expert in variable selection for ML models in tasks of {kwargs["kwargs"]["tarea"]}. 
Analyze these meta-features (TOON format) per column that I extracted with pymfe:
{kwargs["kwargs"]["meta_features_por_columna"]}

Return **only a Python list with the names of the variables** that you recommend for training the model.
Example: ["variable1", "variable2", "variable3"]
No additional text or explanation.
"""

            return prompt
        
        def seleccionar_hiper_parametros():
            prompt = f"""
Act as an expert in ML models from the scikit-learn library. I want you to analyze the following meta-features,
which I obtained using the pymfe library. These are the global meta-features of the dataset. Based on them, and considering
that I am performing a task of {kwargs["kwargs"]["tarea"]} with a {kwargs["kwargs"]["modelo_ml"]} model,
I want you to help me by recommending values for the model’s hyperparameters. Below I provide the
global meta-features and the default hyperparameter configuration so you can help me replace their values.

global meta-features: {kwargs["kwargs"]["meta_features_globales"]}

default hyperparameter configuration: {kwargs["kwargs"]["hiper_parametros_por_defecto"]}

IMPORTANT: Keep in mind that your response must be only a Python dictionary with the names
of the hyperparameters and their recommended values, without any additional text, explanation,
or any format other than a Python dictionary.
Here is an example of how your response should look: {{'hyperparameter_1': value_1, 'hyperparameter_2': value_2}}
"""
            return prompt
        
        def crear_nueva_variable():
            prompt = f"""
Act as an expert in feature engineering for machine learning. I am working with a dataset
of {kwargs["tarea"]} and I need to generate a new variable that can improve the performance of a model.

I will provide you with:
1. The list of dataset columns.
2. A description of the dataset (it may be empty; if so, use only the column names
to infer relationships).

Your task is to generate up to 5 new variables following these rules:

- Return **only** a Python dictionary without any code block (python), explanation,
additional text, or comments. The dictionary rules are:
    - The key is the name of the new variable.
    - The value is a string with the arithmetic operation that defines the new variable using
    the exact names of the provided columns enclosed in backticks.
    This operation must be valid as input for eval() in pandas.
- It is not mandatory to generate 5 variables; they can be between 1 and 5, depending on what
makes sense based on the available information.
- The operations must make sense according to the dataset description and/or
the column names, aiming to create useful features for {kwargs["tarea"]}.
- Avoid generating variables that are trivially identical to existing columns or redundant combinations.
- If you do not find any useful combination or transformation, return an empty dictionary {{}}.

Example:
If the columns were ["unit_price", "quantity"], a possible response could be:
{{"total": "(`unit_price` * `quantity`)"}}

Now, using the information from my dataset, generate the new variables.

Columns: {kwargs["columnas"]}
Description: {kwargs["descripcion"]}
"""
            return prompt

        diccionario_plantilla_prompt = {
            "seleccionar_variables": seleccionar_variables,
            "seleccionar_hiper_parametros": seleccionar_hiper_parametros,
            "crear_nueva_variable": crear_nueva_variable
        }

        return diccionario_plantilla_prompt[plantilla]()

