import os
import json

from datetime import datetime

print_original = print

def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)

class SecuenciaPreprocesamiento:
    _instancia = None
    _inicializado = False

    def __new__(cls, *args, **kwargs):
        if cls._instancia is None:
            cls._instancia = super().__new__(cls)
        return cls._instancia

    def __init__(self):
        if not self._inicializado:
            self.secuencia = {}
            SecuenciaPreprocesamiento._inicializado = True

    def agregar_tecnica(self, fase, tecnica, parametro):
        # print(f"Fase: {fase}, Técnica: {tecnica}, Parámetro: {parametro}\n")
        self.secuencia[fase] = {"tecnica": tecnica, "parametro": parametro}

    def obtener_secuencia(self):
        return self.secuencia
    
    def reiniciar_secuencia(self):
        self.secuencia = {}

    def imprimir_secuencia(self):
        for fase, info in self.secuencia.items():
            print(f"Fase: {fase}, Técnica: {info['tecnica']}, Parámetro: {info['parametro']}\n")

    def guardar_secuencia(self, ruta_archivo="secuencia_preprocesamiento.json"):
        print(f"Guardando secuencia de preprocesamiento en {ruta_archivo}...")
        try:
            ruta_temp = ruta_archivo + ".tmp"
            json_str = json.dumps(self.secuencia, indent=4)

            with open(ruta_temp, "w") as f:
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())

            os.replace(ruta_temp, ruta_archivo)

            print("Secuencia guardada exitosamente.")

        except Exception as e:
            print(f"Error al guardar la secuencia: {e}")
