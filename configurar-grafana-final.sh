#!/bin/bash
echo "🎯 CONFIGURACIÓN GRAFANA - CARACTERES CORREGIDOS"
echo "=============================================="

# Configurar datasource (escapando el !)
echo "📊 Configurando datasource Prometheus..."
curl -X POST "http://admin:HelioBio2025"'!'"@localhost:3002/api/datasources" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"Prometheus-HelioBio",
    "type":"prometheus",
    "url":"http://nn-prometheus:9090",
    "access":"proxy",
    "basicAuth":false,
    "isDefault":true
  }' > /dev/null 2>&1

# Crear dashboard HelioBio
echo "🌍 Creando dashboard HelioBio..."
curl -X POST "http://admin:HelioBio2025"'!'"@localhost:3002/api/dashboards/db" \
  -H "Content-Type: application/json" \
  -d '{
    "dashboard": {
      "id": null,
      "title": "HelioBio Cosmic Monitor",
      "tags": ["chizhevsky", "heliobio"],
      "timezone": "browser",
      "panels": [
        {
          "id": 1,
          "title": "TAC Rate Colectivo",
          "type": "stat",
          "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
          "targets": [{"expr": "tac_rate_colectivo", "refId": "A"}],
          "fieldConfig": {
            "defaults": {
              "color": {"mode": "thresholds"},
              "thresholds": {
                "steps": [
                  {"value": null, "color": "green"},
                  {"value": 1.5, "color": "yellow"},
                  {"value": 2.0, "color": "red"}
                ]
              },
              "unit": "short",
              "decimals": 2
            }
          }
        }
      ],
      "time": {"from": "now-6h", "to": "now"},
      "refresh": "30s"
    },
    "overwrite": true
  }' > /dev/null 2>&1

echo "✅ Configuración completada"
echo "🌐 Accede a: http://localhost:3002"
echo "📊 Busca: HelioBio Cosmic Monitor"
