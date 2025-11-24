"""
carwash_pipeline.py

Pipeline OOP para Inteligencia Comercial del Car Wash.
- Carga datos de lavados
- Construye features para diferentes modelos
- Entrena modelos (regresión, clasificación, forecast, pricing)
- Genera KPIs y tablas para Power BI / PPT / Word
"""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

# Si vas a usar sklearn, statsmodels, prophet, etc:
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.ensemble import RandomForestClassifier


# ======================
# 1. CONFIGURACIÓN
# ======================

@dataclass
class CarWashConfig:
    """Config global del pipeline."""
    path_csv: str = "Base_CarWash.csv"
    sheet_name: str = 0
    date_col: str = "FECHA"
    service_col: str = "SERVICIO_TIPO"
    vehicle_col: str = "VEHICULO_TIPO"
    plate_col: str = "PLACA_TIPO"
    price_col: str = "PRECIO"
    payment_col: str = "TIPO DE PAGO"

    output_folder: str = "output"          # donde dejarás los CSV/XLSX para Power BI
    forecast_horizon_days: int = 30        # horizonte de forecast
    min_visits_for_repeat: int = 2         # umbral para considerar “cliente recurrente”


# ======================
# 2. REPOSITORIO DE DATOS
# ======================

class CarWashDataRepository:
    """Capa que se encarga de leer y preparar la tabla base de servicios."""
    def __init__(self, config: CarWashConfig):
        self.config = config
        self._df_raw: Optional[pd.DataFrame] = None
        self._df_clean: Optional[pd.DataFrame] = None

    def load_raw(self) -> pd.DataFrame:
        if self._df_raw is None:
            df = pd.read_csv(self.config.path_csv) #, sheet_name=self.config.sheet_name)
            #df = pd.read_excel(self.config.path_excel, sheet_name=self.config.sheet_name)
            self._df_raw = df
        return self._df_raw

    def clean(self) -> pd.DataFrame:
        if self._df_clean is None:
            df = self.load_raw().copy()

            # Tipos
            df[self.config.date_col] = pd.to_datetime(df[self.config.date_col])
            df[self.config.price_col] = pd.to_numeric(df[self.config.price_col], errors="coerce")

            # Normalizar texto
            for col in [self.config.service_col,
                        self.config.vehicle_col,
                        self.config.plate_col,
                        self.config.payment_col]:
                df[col] = df[col].astype(str).str.strip().str.upper()

            # Filtrar filas sin precio o fecha
            df = df.dropna(subset=[self.config.date_col, self.config.price_col])

            self._df_clean = df

        return self._df_clean

    def get_base_df(self) -> pd.DataFrame:
        """Atajo semántico."""
        return self.clean()


# ======================
# 3. CONSTRUCTOR DE FEATURES
# ======================

class FeatureBuilder:
    """Genera data sets específicos para cada modelo."""
    def __init__(self, df_base: pd.DataFrame, config: CarWashConfig):
        self.df = df_base.copy()
        self.config = config

        # columnas derivadas comunes
        self.df["dia_semana"] = self.df[self.config.date_col].dt.day_name()
        self.df["dia"] = self.df[self.config.date_col].dt.day
        self.df["mes"] = self.df[self.config.date_col].dt.month

    def build_ticket_features(self) -> pd.DataFrame:
        """
        Dataset para modelo de ticket/ingresos.
        Nivel sugerido: día + servicio + vehículo.
        """
        df = (
            self.df
            .groupby([self.config.date_col,
                      self.config.service_col,
                      self.config.vehicle_col], as_index=False)
            .agg(
                servicios=("PRECIO", "count"),
                ingresos=("PRECIO", "sum"),
                ticket_promedio=("PRECIO", "mean")
            )
        )
        return df

    def build_repeat_features(self) -> pd.DataFrame:
        """
        Dataset a nivel placa: cuántas visitas, ticket medio, mix de servicios, etc.
        """
        df = (
            self.df
            .groupby(self.config.plate_col, as_index=False)
            .agg(
                visitas=("PRECIO", "count"),
                ingresos_total=("PRECIO", "sum"),
                ticket_promedio=("PRECIO", "mean"),
                vehiculos_distintos=(self.config.vehicle_col, "nunique"),
                servicios_distintos=(self.config.service_col, "nunique")
            )
        )
        # etiqueta binaria recurrente vs no recurrente
        df["es_recurrente"] = (df["visitas"] >= self.config.min_visits_for_repeat).astype(int)
        return df

    def build_recommender_features(self) -> pd.DataFrame:
        """
        Dataset a nivel servicio realizado por placa con contexto
        (para recomendar siguiente servicio).
        """
        df = self.df.sort_values([self.config.plate_col, self.config.date_col])
        # aquí podrías crear variables de "último servicio", etc.
        # Por ahora devolvemos la base ordenada.
        return df

    def build_forecast_features(self) -> pd.DataFrame:
        """Serie de tiempo diaria de ingresos y servicios."""
        df = (
            self.df
            .groupby(self.config.date_col, as_index=False)
            .agg(
                ingresos=("PRECIO", "sum"),
                servicios=("PRECIO", "count")
            )
            .set_index(self.config.date_col)
            .asfreq("D")
            .fillna(0)
        )
        return df

    def build_pricing_features(self) -> pd.DataFrame:
        """
        Base para modelo de elasticidad/precio:
        precio, volumen, tipo de servicio, vehículo, día semana, etc.
        """
        df = self.df.copy()
        df["dia_semana"] = df[self.config.date_col].dt.day_name()
        # Podrías agregar columnas de “precio_lista”, “descuento_aplicado”, etc.
        return df


# ======================
# 4. INTERFAZ DE MODELO
# ======================

class BaseModel(ABC):
    """Interface común para todos los modelos de Inteligencia Comercial."""
    def __init__(self, name: str):
        self.name = name
        self.is_trained: bool = False

    @abstractmethod
    def fit(self, df_features: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def predict(self, df_features: Optional[pd.DataFrame] = None) -> Any:
        ...

    @abstractmethod
    def get_kpis(self) -> Dict[str, Any]:
        """Devuelve KPIs clave que irá a Power BI / PPT."""
        ...

    def save_artifacts(self, folder: str) -> None:
        """Opcional: guardar modelo entrenado, parámetros, etc."""
        # aquí podrías usar pickle/joblib
        pass

    def load_artifacts(self, folder: str) -> None:
        """Opcional: cargar modelo entrenado."""
        pass


# ======================
# 5. MODELOS CONCRETOS
# ======================

class TicketRevenueModel(BaseModel):
    """Modelo de regresión para ticket/ingresos (nivel TRÁFICO/VALOR)."""

    def __init__(self):
        super().__init__(name="ticket_revenue")
        self.mean_ticket_: Optional[float] = None

    def fit(self, df_features: pd.DataFrame) -> None:
        # Versión simple: solo guarda el ticket medio (baseline)
        self.mean_ticket_ = df_features["ticket_promedio"].mean()
        self.is_trained = True

    def predict(self, df_features: Optional[pd.DataFrame] = None) -> float:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado.")
        return float(self.mean_ticket_)

    def get_kpis(self) -> Dict[str, Any]:
        return {
            "ticket_promedio_estimado": self.mean_ticket_
        }


class RepeatCustomerModel(BaseModel):
    """Modelo para probabilidad de ser cliente recurrente."""
    def __init__(self):
        super().__init__(name="repeat_customer")
        # aquí iría un clasificador real (logit / random forest)
        self.recurrence_rate_: Optional[float] = None

    def fit(self, df_features: pd.DataFrame) -> None:
        self.recurrence_rate_ = float(df_features["es_recurrente"].mean())
        self.is_trained = True

    def predict(self, df_features: Optional[pd.DataFrame] = None) -> float:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado.")
        # baseline: devuelve probabilidad promedio
        return float(self.recurrence_rate_)

    def get_kpis(self) -> Dict[str, Any]:
        return {
            "probabilidad_media_recurrencia": self.recurrence_rate_
        }


class ServiceRecommenderModel(BaseModel):
    """Modelo para recomendar servicio (Standar / Premium / Salón)."""
    def __init__(self):
        super().__init__(name="service_recommender")

    def fit(self, df_features: pd.DataFrame) -> None:
        # Aquí podrías ajustar un modelo de clasificación / reglas.
        self.is_trained = True

    def predict(self, df_features: Optional[pd.DataFrame] = None) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado.")
        # Placeholder: recomendar siempre PREMIUM
        n = 0 if df_features is None else len(df_features)
        return pd.Series(["PREMIUM"] * n)

    def get_kpis(self) -> Dict[str, Any]:
        return {
            "estrategia_recomendacion": "placeholder: siempre PREMIUM"
        }


class DemandForecastModel(BaseModel):
    """Modelo de forecast de demanda diaria."""
    def __init__(self, horizon_days: int):
        super().__init__(name="demand_forecast")
        self.horizon_days = horizon_days
        self.mean_daily_revenue_: Optional[float] = None

    def fit(self, df_features: pd.DataFrame) -> None:
        self.mean_daily_revenue_ = float(df_features["ingresos"].mean())
        self.is_trained = True

    def predict(self, df_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado.")
        last_date = df_features.index.max()
        future_idx = pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=self.horizon_days,
            freq="D"
        )
        df_forecast = pd.DataFrame(
            {
                "fecha": future_idx,
                "ingresos_forecast": [self.mean_daily_revenue_] * self.horizon_days
            }
        )
        return df_forecast

    def get_kpis(self) -> Dict[str, Any]:
        return {
            "ingresos_diarios_esperados": self.mean_daily_revenue_,
            "horizonte_dias": self.horizon_days
        }


class PriceElasticityModel(BaseModel):
    """Modelo de elasticidad precio-demanda (todavía placeholder)."""

    def __init__(self):
        super().__init__(name="price_elasticity")
        self.elasticity_: Optional[float] = None

    def fit(self, df_features: pd.DataFrame) -> None:
        # Aquí iría una regresión de log(volumen) vs log(precio)
        # Placeholder: elasticidad fija -1.2
        self.elasticity_ = -1.2
        self.is_trained = True

    def predict(self, df_features: Optional[pd.DataFrame] = None) -> float:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado.")
        return float(self.elasticity_)

    def get_kpis(self) -> Dict[str, Any]:
        return {
            "elasticidad_precio": self.elasticity_
        }


# ======================
# 6. PIPELINE ORQUESTADOR
# ======================

class IntelligencePipeline:
    """Orquesta todo el flujo de Inteligencia Comercial."""
    def __init__(self, config: Optional[CarWashConfig] = None):
        self.config = config or CarWashConfig()
        self.repo = CarWashDataRepository(self.config)

        self.df_base: Optional[pd.DataFrame] = None
        self.features: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, BaseModel] = {}
        self.kpis: Dict[str, Dict[str, Any]] = {}

    def build_features(self) -> None:
        self.df_base = self.repo.get_base_df()
        fb = FeatureBuilder(self.df_base, self.config)

        self.features["ticket"] = fb.build_ticket_features()
        self.features["repeat"] = fb.build_repeat_features()
        self.features["recommender"] = fb.build_recommender_features()
        self.features["forecast"] = fb.build_forecast_features()
        self.features["pricing"] = fb.build_pricing_features()

    def init_models(self) -> None:
        self.models["ticket"] = TicketRevenueModel()
        self.models["repeat"] = RepeatCustomerModel()
        self.models["recommender"] = ServiceRecommenderModel()
        self.models["forecast"] = DemandForecastModel(
            horizon_days=self.config.forecast_horizon_days
        )
        self.models["pricing"] = PriceElasticityModel()

    def train_models(self) -> None:
        self.build_features()
        self.init_models()

        self.models["ticket"].fit(self.features["ticket"])
        self.models["repeat"].fit(self.features["repeat"])
        self.models["recommender"].fit(self.features["recommender"])
        self.models["forecast"].fit(self.features["forecast"])
        self.models["pricing"].fit(self.features["pricing"])

        # guardar KPIs
        for key, model in self.models.items():
            self.kpis[key] = model.get_kpis()

    def export_outputs(self) -> None:
        """Ejemplo de exportación a CSV para luego conectarlo en Power BI."""
        import os
        os.makedirs(self.config.output_folder, exist_ok=True)

        # features que quieras mirar en PBI
        self.features["ticket"].to_csv(
            f"{self.config.output_folder}/ticket_features.csv", index=False
        )
        self.features["repeat"].to_csv(
            f"{self.config.output_folder}/repeat_features.csv", index=False
        )

        # forecast
        df_forecast = self.models["forecast"].predict(self.features["forecast"])
        df_forecast.to_csv(
            f"{self.config.output_folder}/forecast_ingresos.csv", index=False
        )

        # KPIs globales
        pd.DataFrame(self.kpis).to_csv(
            f"{self.config.output_folder}/kpi_pack.csv"
        )

    def run_all(self) -> None:
        """Método maestro: entrenar todo y exportar resultados."""
        self.train_models()
        self.export_outputs()


if __name__ == "__main__":
    pipeline = IntelligencePipeline()
    pipeline.run_all()
    print("Pipeline de Inteligencia Comercial Car Wash ejecutado.")
