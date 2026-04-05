import ast
import numpy as np
import pandas as pd
from toon_format import encode

from ..RegistroTecnica import RegistroTecnica
from sklearn.decomposition import PCA
from ..LLM import LLM

class CrearNuevaVariable(RegistroTecnica):
    MODELOS_LLM = {
        "llm": "deepseek-r1:8b",
        "llm_deepseek-r1:8b": "deepseek-r1:8b",
        "llm_llama3.1:8b": "llama3.1:8b",
        "llm_qwen2.5-coder:7b": "qwen2.5-coder:7b",
    }

    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se cree ninguna variable nueva
        semilla: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="crear_nueva_variable")
        self.log_fase = "crear_nueva_variable"
        self.permitir_none = permitir_none
        self.semilla = semilla
        self.config_test = config_test
        self.tarea = ""
        self.descripcion = ""
        self.ALGORITMOS = [
            "llm_deepseek-r1:8b",
            "llm_llama3.1:8b",
            "llm_qwen2.5-coder:7b",
            None
        ]

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Selecciona aleatoriamente la técnica para crear nueva variable
        """
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Aplica la técnica seleccionada para crear una nueva variable
        """
        match self.log_algoritmo:
            case None:
                return X, y
            
            case "llm" | "llm_deepseek-r1:8b" | "llm_llama3.1:8b" | "llm_qwen2.5-coder:7b":
                X_nueva = self._crear_variable_con_llm(X.copy())
                return X_nueva, y
            
            case _:
                raise ValueError(f"Técnica no reconocida: {self.log_algoritmo}")

    def _calcular_parametros(self, X: pd.DataFrame):
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        if self.log_algoritmo is None:
            self.log_params = {}
            self.registrar_parametros(self.log_params)
            return
        
        # Caso LLM: selecciona variables basándose en meta-features
        if self.log_algoritmo not in self.MODELOS_LLM:
            raise ValueError(f"Algoritmo no reconocido: {self.log_algoritmo}")
        
        llm = LLM(self.MODELOS_LLM[self.log_algoritmo])
        prompt = llm.plantillas_prompts(
            "crear_nueva_variable",
            tarea = self.tarea,
            columnas = X.columns.tolist(),
            descripcion = self.descripcion,
        )
        
        try:
            respuesta_llm = llm.generar_respuesta(prompt)
            variables_recomendadas = ast.literal_eval(respuesta_llm)
            if not isinstance(variables_recomendadas, dict):
                raise ValueError("La respuesta del LLM no es un diccionario")
        
        except Exception as e:
            print(f"Error al interpretar la respuesta del LLM o tiempo de espera agotado: {e}")
            raise ValueError(f"{e}")

        
        for var, formula in variables_recomendadas.items():
            self.log_params[var] = formula
        
        self.registrar_parametros(self.log_params)

    def _crear_variable_con_llm(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Crea una nueva variable en el DataFrame X utilizando la fórmula proporcionada por el LLM
        """
        if not self.log_params:
            return X
        
        for var, formula in self.log_params.items():
            try:
                X[var] = X.eval(formula)
            except Exception as e:
                # raise ValueError(f"Error al crear la variable '{var}' con la fórmula '{formula}': {e}")
                continue

        return X