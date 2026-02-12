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
        self.secuencia[fase] = {"tecnica": tecnica, "parametro": parametro}

    def obtener_secuencia(self):
        return self.secuencia
    
    def reiniciar_secuencia(self):
        self.secuencia = {}

    def imprimir_secuencia(self):
        for fase, info in self.secuencia.items():
            print(f"Fase: {fase}, Técnica: {info['tecnica']}, Parámetro: {info['parametro']}\n")
