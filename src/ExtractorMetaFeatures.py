import math
import numpy as np
from pymfe.mfe import MFE

class ExtractorMetaFeatures:
    _GUPOS_META_FEATURES = [
        "landmarking",
        "general",
        "statistical",
        "model-based",
        "info-theory",
        "relative",
        "clustering",
        "complexity",
        "itemset",
        "concept"
    ]
    _CONSTANTE_ERROR = -1111.0
    _CONSTANTE_INFINITO = 2222.0

    def __init__(self):
        pass

    def extraer(self, X, y):
        meta_features = {}

        for grupo in self._GUPOS_META_FEATURES:
            mfe = MFE(groups=[grupo])

            try:
                mfe.fit(X, y)
                ft = mfe.extract()
                meta_features[grupo] = dict(zip(ft[0], ft[1]))

                if len(meta_features[grupo]) == 0:
                    print(f"Seteado grupo: {grupo}")
                    meta_features[grupo] = self._setear_variables_grupo(grupo, self._CONSTANTE_ERROR)
                else:
                    print(f"Completado grupo: {grupo}")

            except Exception as e:
                print(f"ERROR grupo: {grupo}")
                meta_features[grupo] = self._setear_variables_grupo(grupo, self._CONSTANTE_ERROR)
        
        meta_features = self._mapear_meta_features(meta_features)
        meta_features_vectorizadas = self._vectorizar_meta_features(meta_features)

        return meta_features, meta_features_vectorizadas
    
    def _setear_variables_grupo(self, grupo, valor):
        """
        Devuelve un diccionario con todas las meta-features del grupo indicado seteadas
        con el valor indicado.
        
        :param self: Referencia de la instancia de la clase.
        :param grupo: Nombre del grupo de meta-features a setear.
        :param valor: Valor con el que se setearán todas las meta-features del grupo.
        :return: Diccionario con las meta-features del grupo seteadas con el valor indicado.
        """
        grupos_meta_features = {
            "landmarking": {
                "best_node.mean": valor,
                "best_node.sd": valor,
                "elite_nn.mean": valor,
                "elite_nn.sd": valor,
                "linear_discr.mean": valor,
                "linear_discr.sd": valor,
                "naive_bayes.mean": valor,
                "naive_bayes.sd": valor,
                "one_nn.mean": valor,
                "one_nn.sd": valor,
                "random_node.mean": valor,
                "random_node.sd": valor,
                "worst_node.mean": valor,
                "worst_node.sd": valor
            },
            "general": {
                "attr_to_inst": valor,
                "cat_to_num": valor,
                "freq_class.mean": valor,
                "freq_class.sd": valor,
                "inst_to_attr": valor,
                "nr_attr": valor,
                "nr_bin": valor,
                "nr_cat": valor,
                "nr_class": valor,
                "nr_inst": valor,
                "nr_num": valor,
                "num_to_cat": valor
            },
            "statistical": {
                "can_cor.mean": valor,
                "can_cor.sd": valor,
                "cor.mean": valor,
                "cor.sd": valor,
                "cov.mean": valor,
                "cov.sd": valor,
                "eigenvalues.mean": valor,
                "eigenvalues.sd": valor,
                "g_mean.mean": valor,
                "g_mean.sd": valor,
                "gravity": valor,
                "h_mean.mean": valor,
                "h_mean.sd": valor,
                "iq_range.mean": valor,
                "iq_range.sd": valor,
                "kurtosis.mean": valor,
                "kurtosis.sd": valor,
                "lh_trace": valor,
                "mad.mean": valor,
                "mad.sd": valor,
                "max.mean": valor,
                "max.sd": valor,
                "mean.mean": valor,
                "mean.sd": valor,
                "median.mean": valor,
                "median.sd": valor,
                "min.mean": valor,
                "min.sd": valor,
                "nr_cor_attr": valor,
                "nr_disc": valor,
                "nr_norm": valor,
                "nr_outliers": valor,
                "p_trace": valor,
                "range.mean": valor,
                "range.sd": valor,
                "roy_root": valor,
                "sd.mean": valor,
                "sd.sd": valor,
                "sd_ratio": valor,
                "skewness.mean": valor,
                "skewness.sd": valor,
                "sparsity.mean": valor,
                "sparsity.sd": valor,
                "t_mean.mean": valor,
                "t_mean.sd": valor,
                "var.mean": valor,
                "var.sd": valor,
                "w_lambda": valor
            },
            "model-based": {
                "leaves": valor,
                "leaves_branch.mean": valor,
                "leaves_branch.sd": valor,
                "leaves_corrob.mean": valor,
                "leaves_corrob.sd": valor,
                "leaves_homo.mean": valor,
                "leaves_homo.sd": valor,
                "leaves_per_class.mean": valor,
                "leaves_per_class.sd": valor,
                "nodes": valor,
                "nodes_per_attr": valor,
                "nodes_per_inst": valor,
                "nodes_per_level.mean": valor,
                "nodes_per_level.sd": valor,
                "nodes_repeated.mean": valor,
                "nodes_repeated.sd": valor,
                "tree_depth.mean": valor,
                "tree_depth.sd": valor,
                "tree_imbalance.mean": valor,
                "tree_imbalance.sd": valor,
                "tree_shape.mean": valor,
                "tree_shape.sd": valor,
                "var_importance.mean": valor,
                "var_importance.sd": valor
            },
            "info-theory": {
                "attr_conc.mean": valor,
                "attr_conc.sd": valor,
                "attr_ent.mean": valor,
                "attr_ent.sd": valor,
                "class_conc.mean": valor,
                "class_conc.sd": valor,
                "class_ent": valor,
                "eq_num_attr": valor,
                "joint_ent.mean": valor,
                "joint_ent.sd": valor,
                "mut_inf.mean": valor,
                "mut_inf.sd": valor,
                "ns_ratio": valor
            },
            "relative": {
                "best_node.mean.relative": valor,
                "best_node.sd.relative": valor,
                "elite_nn.mean.relative": valor,
                "elite_nn.sd.relative": valor,
                "linear_discr.mean.relative": valor,
                "linear_discr.sd.relative": valor,
                "naive_bayes.mean.relative": valor,
                "naive_bayes.sd.relative": valor,
                "one_nn.mean.relative": valor,
                "one_nn.sd.relative": valor,
                "random_node.mean.relative": valor,
                "random_node.sd.relative": valor,
                "worst_node.mean.relative": valor,
                "worst_node.sd.relative": valor
            },
            "clustering": {
                "ch": valor,
                "int": valor,
                "nre": valor,
                "pb": valor,
                "sc": valor,
                "sil": valor,
                "vdb": valor,
                "vdu": valor
            },
            "complexity": {
                "c1": valor,
                "c2": valor,
                "cls_coef": valor,
                "density": valor,
                "f1.mean": valor,
                "f1.sd": valor,
                "f1v.mean": valor,
                "f1v.sd": valor,
                "f2.mean": valor,
                "f2.sd": valor,
                "f3.mean": valor,
                "f3.sd": valor,
                "f4.mean": valor,
                "f4.sd": valor,
                "hubs.mean": valor,
                "hubs.sd": valor,
                "l1.mean": valor,
                "l1.sd": valor,
                "l2.mean": valor,
                "l2.sd": valor,
                "l3.mean": valor,
                "l3.sd": valor,
                "lsc": valor,
                "n1": valor,
                "n2.mean": valor,
                "n2.sd": valor,
                "n3.mean": valor,
                "n3.sd": valor,
                "n4.mean": valor,
                "n4.sd": valor,
                "t1.mean": valor,
                "t1.sd": valor,
                "t2": valor,
                "t3": valor,
                "t4": valor
            },
            "itemset": {
                "one_itemset.mean": valor,
                "one_itemset.sd": valor,
                "two_itemset.mean": valor,
                "two_itemset.sd": valor
            },
            "concept": {
                "cohesiveness.mean": valor,
                "cohesiveness.sd": valor,
                "conceptvar.mean": valor,
                "conceptvar.sd": valor,
                "impconceptvar.mean": valor,
                "impconceptvar.sd": valor,
                "wg_dist.mean": valor,
                "wg_dist.sd": valor
            }
        }

        return grupos_meta_features.get(grupo, {})


    def _mapear_meta_features(self, meta_features: dict):
        """
        Mapea las meta-features extraídas a tipos nativos de Python y reemplaza NaN / inf por valores específicos.
        
        :param self: Referencia de la instancia de la clase.
        :param meta_features: Diccionario con las meta-features a mapear.
        :type meta_features: dict
        :return: Diccionario con las meta-features mapeadas a valores nativos de Python.
        :rtype: dict
        """
        meta_features_mapeadas = {}
        for grupo, features in meta_features.items():
            meta_features_mapeadas[grupo] = {}
            for meta_feature, valor in features.items():
                if valor is None or (isinstance(valor, float) and (math.isnan(valor))):
                    valor = self._CONSTANTE_ERROR

                if isinstance(valor, float) and math.isinf(valor):
                    valor = self._CONSTANTE_INFINITO

                if isinstance(valor, np.float64):
                    valor = float(valor)

                if isinstance(valor, np.int64) or isinstance(valor, int):
                    valor = float(valor)

                meta_features_mapeadas[grupo][meta_feature] = valor
        
        return meta_features_mapeadas
                

    def _vectorizar_meta_features(self, meta_features: dict):
        """
        Vectoriza las meta-features extraídas en una lista, manteniendo el orden
        de los grupos y de las meta-features.
        
        :param self: Referencia de la instancia de la clase.
        :param meta_features: Diccionario con las meta-features a vectorizar.
        :type meta_features: dict
        :return: Lista con los valores de las meta-features vectorizadas.
        :rtype: list
        """
        print(meta_features)
        print("\n\n")
        meta_features_vectorizadas = [valor for subdict in meta_features.values() for valor in subdict.values()]

        
        print(meta_features_vectorizadas)
        return meta_features_vectorizadas


    def _agregar_meta_features_personalizadas(self, meta_features: dict):
        """
        Agrega meta-features personalizadas que no se encuentran en la librería pymfe.
        Las nuevas meta-features se agregarán al grupo "personalizadas" con el prefijo
        "personalizado_".
        
        :param self: Referencia de la instancia de la clase.
        :param meta_features: Diccionario con las meta-features al que se agregarán las personalizadas.
        :type meta_features: dict
        :return: Diccionario con las meta-features incluyendo las personalizadas.
        :rtype: dict
        """
        pass