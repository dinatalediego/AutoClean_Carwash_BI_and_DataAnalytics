"""
generate_reports.py

Orquesta:
- Pipeline numérico
- Capa económica
- Reportes PPTX, PDF, Script, Correo
"""

from carwash_pipeline import IntelligencePipeline
from carwash_econ import CarWashEconEngine
from carwash_reporting import (
    ReportingContext,
    PPTXReporter,
    PDFReporter,
    TextReporter,
)


def main():
    # 1. Ejecutar pipeline de inteligencia
    pipeline = IntelligencePipeline()
    pipeline.run_all()

    # 2. Capa económica (equilibrio + elasticidad)
    econ_engine = CarWashEconEngine(pipeline)
    eq_summary = econ_engine.analyze_equilibrium()

    try:
        elasticity_kpis = econ_engine.fit_price_elasticity()
    except Exception:
        elasticity_kpis = {}

    narrative_builder = econ_engine.build_narrative()
    equilibrium_bullets = narrative_builder.build_equilibrium_story()
    elasticity_bullets = narrative_builder.build_elasticity_story()

    # 3. Contexto compartido de reporting
    ctx = ReportingContext(
        kpis=pipeline.kpis,
        equilibrium_bullets=equilibrium_bullets,
        elasticity_bullets=elasticity_bullets,
        output_folder=pipeline.config.output_folder,
    )

    # 4. PPTX
    pptx_reporter = PPTXReporter(ctx)
    pptx_path = pptx_reporter.save("Reporte_Inteligencia_Comercial_CarWash.pptx")
    print("PPTX generado:", pptx_path)

    # 5. PDF
    pdf_reporter = PDFReporter(ctx)
    pdf_path = pdf_reporter.save("Informe_Ejecutivo_CarWash.pdf")
    print("PDF generado:", pdf_path)

    # 6. Textos (script verbal + correo)
    text_reporter = TextReporter(ctx)
    text_paths = text_reporter.save_texts(
        script_filename="Script_Verbal_CarWash.txt",
        correo_filename="Correo_Propuesta_CarWash.txt",
    )
    print("Textos generados:", text_paths)


if __name__ == "__main__":
    main()
