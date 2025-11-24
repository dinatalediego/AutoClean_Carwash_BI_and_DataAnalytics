"""
carwash_econ.py

Capa de Microeconomía + Econometría OOP para el Car Wash,
construida encima de carwash_pipeline.IntelligencePipeline.

- Resume el equilibrio de demanda (media, varianza, picos, valles)
- Permite correr regresiones sencillas (si tienes statsmodels instalado)
- Construye narrativa económica (storytelling) lista para usar en PPT / clases
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

# Opcional: si quieres usar statsmodels para regresiones económicas:
try:
    import statsmodels.api as sm
except ImportError:
    sm = None


# =========================
# 1. RESUMEN DE EQUILIBRIO
# =========================

@dataclass
class EquilibriumSummary:
    """Resumen del 'equilibrio empírico' de la demanda diaria."""
    mean_revenue: float
    std_revenue: float
    min_revenue: float
    max_revenue: float
    peak_days: List[pd.Timestamp]
    trough_days: List[pd.Timestamp]

    def to_bullets(self) -> List[str]:
        """Devuelve bullets tipo storytelling."""
        bullets = [
            f"La demanda diaria se estabiliza alrededor de S/ {self.mean_revenue:,.0f} con una desviación de S/ {self.std_revenue:,.0f}.",
            f"El mínimo observado fue S/ {self.min_revenue:,.0f} y el máximo S/ {self.max_revenue:,.0f}.",
        ]
        if self.peak_days:
            peaks = ", ".join(d.strftime("%d-%b") for d in self.peak_days)
            bullets.append(f"Los picos de demanda se concentran en los días: {peaks}.")
        if self.trough_days:
            troughs = ", ".join(d.strftime("%d-%b") for d in self.trough_days)
            bullets.append(f"Los valles de demanda aparecen en: {troughs}.")
        return bullets


class DemandEquilibriumAnalyzer:
    """
    Analiza la serie diaria de ingresos para identificar:
    - media (equilibrio empírico)
    - dispersión
    - picos y valles según umbrales relativos
    """

    def __init__(self, daily_revenue: pd.Series):
        """
        Parameters
        ----------
        daily_revenue : pd.Series
            Serie con índice fecha y valores de ingresos diarios.
        """
        if not isinstance(daily_revenue.index, pd.DatetimeIndex):
            raise ValueError("daily_revenue debe tener índice DatetimeIndex.")
        self.series = daily_revenue.sort_index()

    def compute_equilibrium(
        self,
        low_threshold: float = 0.75,
        high_threshold: float = 1.25
    ) -> EquilibriumSummary:
        """Calcula media, varianza y detecta picos/valles relativos."""
        mean_rev = float(self.series.mean())
        std_rev = float(self.series.std(ddof=1))
        min_rev = float(self.series.min())
        max_rev = float(self.series.max())

        low_cut = mean_rev * low_threshold
        high_cut = mean_rev * high_threshold

        trough_days = self.series[self.series < low_cut].index.to_list()
        peak_days = self.series[self.series > high_cut].index.to_list()

        return EquilibriumSummary(
            mean_revenue=mean_rev,
            std_revenue=std_rev,
            min_revenue=min_rev,
            max_revenue=max_rev,
            peak_days=peak_days,
            trough_days=trough_days,
        )


# =========================
# 2. MODELOS ECONOMÉTRICOS
# =========================

class DailyRevenueRegression:
    """
    Regresión sencilla de ingresos diarios contra dummies de día de semana
    y otras variables agregadas. Uso didáctico/económico.
    """

    def __init__(self):
        self.result = None

    def fit(self, daily_df: pd.DataFrame) -> None:
        """
        Espera un DataFrame con columnas:
        - 'ingresos'
        - opcional: 'servicios', etc.
        Índice debe ser fecha.
        """
        if sm is None:
            raise ImportError("statsmodels no está instalado. Instálalo para usar DailyRevenueRegression.")

        df = daily_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("daily_df debe tener índice DatetimeIndex.")

        df["dow"] = df.index.dayofweek  # 0=lunes ... 6=domingo
        dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)

        X = pd.concat([dummies], axis=1)
        X = sm.add_constant(X)
        y = df["ingresos"]

        model = sm.OLS(y, X)
        self.result = model.fit()

    def summary_text(self, max_lines: int = 20) -> str:
        """Devuelve un resumen de la regresión para pegarlo en anexos / reports."""
        if self.result is None:
            return "Modelo no ha sido ajustado aún."
        s = self.result.summary().as_text()
        # recorta para no devolver algo exageradamente largo
        lines = s.splitlines()
        return "\n".join(lines[:max_lines])


class PriceElasticityModel:
    """
    Modelo simple de elasticidad precio-demanda: log(Q) ~ log(P) + controles.
    Necesita que tengas columnas de 'precio' y 'cantidad' a nivel de observación.
    """

    def __init__(self):
        self.result = None
        self.elasticity_: Optional[float] = None

    def fit(self, df: pd.DataFrame, price_col: str, qty_col: str) -> None:
        if sm is None:
            raise ImportError("statsmodels no está instalado. Instálalo para usar PriceElasticityModel.")

        df = df.copy()
        df = df[[price_col, qty_col]].dropna()

        df["log_p"] = np.log(df[price_col])
        df["log_q"] = np.log(df[qty_col])

        X = sm.add_constant(df["log_p"])
        y = df["log_q"]

        model = sm.OLS(y, X)
        self.result = model.fit()

        # coef de log_p es la elasticidad
        self.elasticity_ = float(self.result.params["log_p"])

    def get_kpis(self) -> Dict[str, Any]:
        if self.elasticity_ is None:
            return {"elasticidad_precio": None}
        return {"elasticidad_precio": self.elasticity_}

    def summary_text(self, max_lines: int = 20) -> str:
        if self.result is None:
            return "Modelo no ha sido ajustado aún."
        s = self.result.summary().as_text()
        lines = s.splitlines()
        return "\n".join(lines[:max_lines])


# =========================
# 3. NARRATIVA ECONÓMICA
# =========================

class EconNarrativeBuilder:
    """
    Toma el EquilibriumSummary y (opcionalmente) resultados de regresión
    para construir narrativa tipo masterclass / comité.
    """

    def __init__(
        self,
        equilibrium: EquilibriumSummary,
        elasticity_kpis: Optional[Dict[str, Any]] = None
    ):
        self.eq = equilibrium
        self.elasticity_kpis = elasticity_kpis or {}

    def build_equilibrium_story(self) -> List[str]:
        """
        Storytelling base: 'sin intervención, la demanda se estabiliza...'
        """
        bullets = self.eq.to_bullets()

        extra = [
            "Interpretación microeconómica: el sistema se encuentra en un equilibrio empírico 'de hecho', no necesariamente óptimo.",
            "La presencia de picos y valles muestra que la disposición a pagar y el uso del servicio varían entre días.",
            "Los picos suelen concentrarse donde la utilidad marginal del lavado es alta (clima, fin de semana, acumulación de suciedad).",
            "Los valles representan días donde la utilidad marginal es baja y solo una política de precios/promos puede elevar la demanda."
        ]
        bullets.extend(extra)
        return bullets

    def build_elasticity_story(self) -> List[str]:
        """
        Explica la elasticidad estimada, si está disponible.
        """
        bullets: List[str] = []
        e = self.elasticity_kpis.get("elasticidad_precio", None)
        if e is None:
            bullets.append("Aún no se ha estimado la elasticidad precio-demanda con datos de variación de precios.")
            return bullets

        bullets.append(f"La elasticidad precio-demanda estimada es de aproximadamente {e:0.2f}.")
        if e < -1:
            bullets.append(
                "Esto implica una demanda elástica: un aumento porcentual en el precio reduce el volumen en mayor proporción."
            )
            bullets.append(
                "En estos casos, subir precios debe hacerse con cuidado en segmentos sensibles, pero puede funcionar en segmentos poco sensibles (camionetas, servicios premium)."
            )
        elif -1 <= e < 0:
            bullets.append(
                "Esto sugiere una demanda inelástica: hay espacio para subir precios sin una caída fuerte en volumen."
            )
        else:
            bullets.append(
                "La elasticidad estimada es inusual (cercana a cero o positiva); puede indicar problemas de datos o segmentación."
            )

        return bullets

    def build_masterclass_script(self) -> str:
        """
        Devuelve un texto corrido que podrías usar como guion en una masterclass.
        """
        parts: List[str] = []

        parts.append(
            "Sin intervención externa, la demanda diaria del Car Wash se estabiliza alrededor de un nivel de equilibrio empírico."
        )
        parts.extend(self.build_equilibrium_story())

        if self.elasticity_kpis:
            parts.append("")
            parts.append("Desde el punto de vista de la teoría, la elasticidad precio-demanda nos dice cómo se mueve este equilibrio.")
            parts.extend(self.build_elasticity_story())

        return "\n- ".join([""] + parts)


# =========================
# 4. ENGINE CONECTADO AL PIPELINE
# =========================

class CarWashEconEngine:
    """
    Capa que se conecta a IntelligencePipeline (carwash_pipeline)
    y genera resúmenes económicos / narrativas.
    """

    def __init__(self, pipeline: "IntelligencePipeline"):  # type: ignore
        self.pipeline = pipeline
        self.equilibrium_summary: Optional[EquilibriumSummary] = None
        self.elasticity_model: Optional[PriceElasticityModel] = None
        self.elasticity_kpis: Dict[str, Any] = {}

    def analyze_equilibrium(self) -> EquilibriumSummary:
        """
        Usa la tabla forecast_features (o la serie de ingresos diarios base)
        para calcular el equilibrio empírico.
        """
        # Usamos los features construidos por el pipeline:
        daily_df = self.pipeline.features.get("forecast")
        if daily_df is None:
            raise RuntimeError("El pipeline no tiene 'forecast' features disponibles.")

        # daily_df: index=fecha, columnas ['ingresos','servicios']
        analyzer = DemandEquilibriumAnalyzer(daily_df["ingresos"])
        self.equilibrium_summary = analyzer.compute_equilibrium()
        return self.equilibrium_summary

    def fit_price_elasticity(self, df_pricing: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Ajusta el modelo de elasticidad si tienes datos de precios y cantidades.
        df_pricing: opcional, si no se pasa usa pipeline.features['pricing'].
        """
        if df_pricing is None:
            df_pricing = self.pipeline.features.get("pricing")
        if df_pricing is None:
            raise RuntimeError("No hay datos de pricing disponibles para estimar elasticidad.")

        if "PRECIO" not in df_pricing.columns:
            raise ValueError("Se espera una columna 'PRECIO' en df_pricing.")
        # cantidad = 1 por servicio; si agregas cantidades explícitas, cámbialo.
        df_pricing = df_pricing.copy()
        df_pricing["cantidad"] = 1.0

        model = PriceElasticityModel()
        model.fit(df_pricing, price_col="PRECIO", qty_col="cantidad")
        self.elasticity_model = model
        self.elasticity_kpis = model.get_kpis()
        return self.elasticity_kpis

    def build_narrative(self) -> EconNarrativeBuilder:
        """
        Construye el builder de narrativa usando equilibrio + elasticidad (si existe).
        """
        if self.equilibrium_summary is None:
            raise RuntimeError("Primero debes calcular el equilibrio con analyze_equilibrium().")
        return EconNarrativeBuilder(self.equilibrium_summary, self.elasticity_kpis)
