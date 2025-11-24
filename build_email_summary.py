# build_email_summary.py
import pandas as pd
from pathlib import Path

def main():
    base = Path("output")
    kpi_path = base / "kpi_pack.csv"
    if not kpi_path.exists():
        print("No se encontró kpi_pack.csv")
        return

    df = pd.read_csv(kpi_path)
    # Ajusta esto a tu estructura real de kpi_pack.csv
    # Ejemplo simple: mostrar columnas y primera fila
    print("Resumen automático del pipeline Car Wash\n")
    print("Columnas del kpi_pack:")
    print(", ".join(df.columns))
    print("\nPrimera fila de KPIs:")
    print(df.head(1).to_string(index=False))

if __name__ == "__main__":
    main()
