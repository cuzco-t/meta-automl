class SelectorHiperParametros:
    def __init__(self, modelo, parametros):
        self.modelo = modelo
        self.parametros = parametros

    def selecionar(self, X_treino, y_treino):
        from sklearn.model_selection import GridSearchCV

        grid_search = GridSearchCV(estimator=self.modelo, param_grid=self.parametros, cv=5)
        grid_search.fit(X_treino, y_treino)

        return grid_search.best_estimator_