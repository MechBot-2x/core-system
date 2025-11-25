#!/bin/bash
echo "🔮 MONITOREO HELIOBIO ACTIVO - $(date)"
echo "=================================="

while true; do
    # Ejecutar análisis TAC Rate
    RESULT=$(python3 heliobio/social/analisis_tac_rate.py)
    TAC_RATE=$(echo "$RESULT" | grep tac_rate | cut -d'"' -f4)
    ESTADO=$(echo "$RESULT" | grep estado | cut -d'"' -f4)

    echo "🌍 TAC Rate: $TAC_RATE | Estado: $ESTADO | $(date +%H:%M:%S)"

    # Alertas automáticas
    if (( $(echo "$TAC_RATE > 2.0" | bc -l) )); then
        echo "🚨 ALERTA: TAC Rate elevado - Posible evento social inminente"
    fi

    sleep 30
done
