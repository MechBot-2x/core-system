#!/bin/bash
echo "🔮 MONITOREO HELIOBIO CORREGIDO - $(date)"
echo "========================================"

while true; do
    RESULT=$(python3 heliobio/social/analisis_tac_rate.py 2>/dev/null | tr -d '\000-\031')
    TAC_RATE=$(echo "$RESULT" | grep tac_rate | sed 's/.*"tac_rate": \([0-9.]*\).*/\1/')
    ESTADO=$(echo "$RESULT" | grep estado | sed 's/.*"estado": "\([^"]*\)".*/\1/')

    if [ -n "$TAC_RATE" ]; then
        echo "🌍 TAC Rate: $TAC_RATE | Estado: $ESTADO | $(date +%H:%M:%S)"

        # Alertas simples sin bc
        if (( $(echo "$TAC_RATE > 2.0" | bc -l 2>/dev/null || echo "0") )); then
            echo "🚨 ALERTA: TAC Rate elevado"
        fi
    else
        echo "⏳ Esperando datos TAC Rate..."
    fi

    sleep 30
done
