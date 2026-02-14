from .SecuenciaPreprocesamiento import SecuenciaPreprocesamiento

class RegistroTecnica:
    def __init__(self):
        self.nombre_fase = None
        self.tecnica_seleccionada_ = None
        self.parametro_tecnica_ = None

    def registrar_tecnica(self, fase, tecnica, parametro):
        secuencia = SecuenciaPreprocesamiento()
        secuencia.agregar_tecnica(fase, tecnica, parametro)

    def imprimir_fase(self, fase):
        secuencia = SecuenciaPreprocesamiento()
        print(secuencia.secuencia[fase])