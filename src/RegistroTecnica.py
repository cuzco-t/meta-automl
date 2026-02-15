from .SecuenciaPreprocesamiento import SecuenciaPreprocesamiento

class RegistroTecnica:
    def __init__(self):
        self.log_fase = None
        self.log_algoritmo = None
        self.log_params = None

    def registrar_tecnica(self, fase, tecnica, parametro):
        secuencia = SecuenciaPreprocesamiento()
        secuencia.agregar_tecnica(fase, tecnica, parametro)

    def imprimir_fase(self, fase):
        secuencia = SecuenciaPreprocesamiento()
        print(secuencia.secuencia[fase])