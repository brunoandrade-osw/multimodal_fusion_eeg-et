"""
Ensemble de Gradient Boosting do modelo de eye-tracking (nb 06, Seção 4).

Em vez de escolher UMA configuração de hiperparâmetros (o que, com N=41, é ruidoso — ver
reports/ml/08_fase6/), o modelo é a média de P(TEA) de todas as GB de uma grade fixa:

    k (features pelo F da ANOVA) × max_depth × learning_rate × subsample × n_estimators
    {5,8,10,15,20}             × {1,2,3}   × {0.03,0.1,0.3} × {1.0,0.7} × {50,100,200,400} = 360

Tudo que aprende com os dados (escala, filtro de variância, ranking F, árvores) é ajustado só no
`fit`, então dentro de um CV nada do fold de teste vaza.
"""
import copy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import RobustScaler


def log_t(X):
    return np.log1p(np.abs(X)) * np.sign(X)


class EnsembleGB(ClassifierMixin, BaseEstimator):
    def __init__(self, ks=(5, 8, 10, 15, 20), depths=(1, 2, 3), lrs=(0.03, 0.1, 0.3),
                 subsamples=(1.0, 0.7), n_estimators=(50, 100, 200, 400), random_state=42):
        self.ks, self.depths, self.lrs, self.subsamples = ks, depths, lrs, subsamples
        self.n_estimators, self.random_state = n_estimators, random_state

    def _z(self, X):
        return self.scaler_.transform(log_t(np.asarray(X, float)))[:, self.ok_]

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.scaler_ = RobustScaler().fit(log_t(X))
        self.ok_ = self.scaler_.transform(log_t(X)).var(0) > 0
        Z = self._z(X)
        self.f_ = np.nan_to_num(f_classif(Z, y)[0], nan=-1)
        self.ordem_ = np.argsort(-self.f_, kind='stable')          # ranking F, só do treino
        self.gbs_ = []                                             # (colunas, GB com max(n_estimators))
        for k in self.ks:
            c = self.ordem_[:k]
            for d in self.depths:
                for lr in self.lrs:
                    for ss in self.subsamples:
                        gb = GradientBoostingClassifier(n_estimators=max(self.n_estimators), learning_rate=lr,
                                                        max_depth=d, subsample=ss,
                                                        random_state=self.random_state).fit(Z[:, c], y)
                        self.gbs_.append((c, gb))
        return self

    def predict_proba(self, X):
        Z = self._z(X)
        p, m = np.zeros(len(Z)), 0
        for c, gb in self.gbs_:                                    # cada GB rende 1 membro por n_estimators
            for j, pj in enumerate(gb.staged_predict_proba(Z[:, c]), 1):
                if j in self.n_estimators:
                    p += pj[:, 1]; m += 1
        p /= m
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]

    # ── interpretação (usar só no refit com todos os sujeitos) ──
    def colunas_usadas(self):
        """Índices (no X original) das max(ks) features que entram em algum membro, em ordem de F."""
        return np.where(self.ok_)[0][self.ordem_[:max(self.ks)]]

    def membros(self):
        """Gera (posições dentro de colunas_usadas(), GB truncada em n árvores) para os 360 membros."""
        for c, gb in self.gbs_:
            for n in self.n_estimators:
                g = copy.copy(gb)
                g.estimators_, g.n_estimators_, g.n_estimators = gb.estimators_[:n], n, n
                yield np.arange(len(c)), g

    def importancias(self):
        """Importância média (impureza) por feature de colunas_usadas(), entre os 360 membros."""
        imp, m = np.zeros(max(self.ks)), 0
        for pos, g in self.membros():
            imp[pos] += g.feature_importances_; m += 1
        return imp / m

    def shap_values(self, X):
        """SHAP (log-odds) médio entre os 360 membros; colunas = colunas_usadas()."""
        import shap
        Z = self._z(X)[:, self.ordem_[:max(self.ks)]]
        sv, m = np.zeros(Z.shape), 0
        for pos, g in self.membros():
            sv[:, pos] += shap.TreeExplainer(g).shap_values(Z[:, pos]); m += 1
        return sv / m, Z
