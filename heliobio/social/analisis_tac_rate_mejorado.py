#!/usr/bin/env python3
"""
ANALIZADOR TAC RATE MEJORADO - Con datos dinámicos y exportación Prometheus
"""

import random
import time
import json
from datetime import datetime

class TacRateAvanzado:
    def __init__(self):
        self.version = "Chizhevsky-CoreSystem-v2.0"

    def generar_datos_reales(self):
        """Genera datos TAC Rate más realistas con variación"""
        # Base + variación aleatoria + tendencia temporal
        base = 1.5
        variacion = random.uniform(-0.3, 0.5)

        # Efecto hora del día (mayor actividad en horas pico)
        hora = datetime.now().hour
        if 8 <= hora <= 10 or 18 <= hora <= 20:
            variacion += 0.2

        tac_rate = max(0.8, min(3.0, base + variacion))

        # Determinar estado basado en TAC Rate
        if tac_rate < 1.2:
            estado = "✅ CALMA COLECTIVA"
        elif tac_rate < 1.8:
            estado = "⚡ ACTIVIDAD NORMAL"
        elif tac_rate < 2.2:
            estado = "⚠️  ALTERACIÓN MODERADA"
        else:
            estado = "🚨 ALTA PERTURBACIÓN"

        return {
            'tac_rate': round(tac_rate, 3),
            'estado': estado,
            'timestamp': datetime.now().isoformat(),
            'hora_local': datetime.now().strftime("%H:%M:%S"),
            'version': self.version,
            'metricas_adicionales': {
                'volatilidad': round(random.uniform(0.1, 0.9), 2),
                'cohesion_social': round(random.uniform(0.3, 0.95), 2),
                'estres_colectivo': round(random.uniform(0.2, 0.8), 2)
            }
        }

    def exportar_prometheus(self, datos):
        """Formato para Prometheus"""
        return f"""
# HELP tac_rate_colectivo Tasa de Alteración Colectiva
# TYPE tac_rate_colectivo gauge
tac_rate_colectivo {datos['tac_rate']}

# HELP cohesion_social Cohesión Social Colectiva
# TYPE cohesion_social gauge
cohesion_social {datos['metricas_adicionales']['cohesion_social']}

# HELP estres_colectivo Estrés Colectivo
# TYPE estres_colectivo gauge
estres_colectivo {datos['metricas_adicionales']['estres_colectivo']}
"""

if __name__ == "__main__":
    print("🔮 TAC RATE AVANZADO - Sistema HelioBio Mejorado")
    print("==============================================")

    analyzer = TacRateAvanzado()

    while True:
        datos = analyzer.generar_datos_reales()

        # Mostrar en consola
        print(f"🌍 TAC Rate: {datos['tac_rate']} | Estado: {datos['estado']} | Hora: {datos['hora_local']}")

        # Exportar para Prometheus (podría escribir a archivo)
        metrics = analyzer.exportar_prometheus(datos)

        # Esperar 30 segundos
        time.sleep(30)
