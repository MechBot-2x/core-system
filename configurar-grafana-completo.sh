#!/bin/bash
echo "🎯 CONFIGURACIÓN COMPLETA GRAFANA DESDE CONSOLA"
echo "=============================================="

# Configurar datasource Prometheus
echo "📊 Configurando datasource Prometheus..."
curl -X POST "http://admin:HelioBio2025!@localhost:3002/api/datasources" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"Prometheus",
    "type":"prometheus",
    "url":"http://nn-prometheus:9090",
    "access":"proxy",
    "basicAuth":false,
    "isDefault":true
  }' > /dev/null 2>&1 && echo "✅ Prometheus configurado"

# Crear dashboard HelioBio
echo "🌍 Creando dashboard HelioBio..."
curl -X POST "http://admin:HelioBio2025!@localhost:3002/api/dashboards/db" \
  -H "Content-Type: application/json" \
  -d '{
    "dashboard": {
      "title": "🌍 HelioBio Cosmic Monitor",
      "tags": ["chizhevsky", "heliobio"],
      "panels": [
        {
          "id": 1,
          "title": "TAC Rate Colectivo",
          "type": "stat",
          "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
          "targets": [{"expr": "tac_rate_colectivo", "refId": "A"}]
        }
      ],
      "time": {"from": "now-6h", "to": "now"}
    },
    "overwrite": true
  }' > /dev/null 2>&1 && echo "✅ Dashboard creado"

echo ""
echo "🎉 CONFIGURACIÓN COMPLETADA"
echo "=========================="
echo "🌐 Accede a: http://localhost:3002"
echo "📊 Dashboard: 🌍 HelioBio Cosmic Monitor"
echo "👤 Usuario: admin"
echo "🔑 Password: HelioBio2025!"
