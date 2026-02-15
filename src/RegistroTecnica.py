from .SecuenciaPreprocesamiento import SecuenciaPreprocesamiento

class RegistroTecnica:
    def __init__(self, log_fase=None, log_algoritmo=None, log_params=None):
        self.log_fase = log_fase
        self.log_algoritmo = log_algoritmo
        self.log_params = log_params

    def registrar_tecnica(self, fase, tecnica, parametro):
        secuencia = SecuenciaPreprocesamiento()
        secuencia.agregar_tecnica(fase, tecnica, parametro)

    def registrar_algoritmo(self, algoritmo):
        secuencia = SecuenciaPreprocesamiento()
        secuencia.agregar_tecnica(self.log_fase, algoritmo, self.log_params)

    def registrar_parametros(self, parametros):
        secuencia = SecuenciaPreprocesamiento()
        secuencia.agregar_tecnica(self.log_fase, self.log_algoritmo, parametros)

    def imprimir_fase(self, fase):
        secuencia = SecuenciaPreprocesamiento()
        print(secuencia.secuencia[fase])