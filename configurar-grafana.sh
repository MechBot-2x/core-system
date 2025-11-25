#!/bin/bash
echo "🎯 Configurando Grafana automáticamente..."

# Esperar que Grafana esté listo
sleep 10

# Configurar data source via API
curl -X POST "http://localhost:3002/api/datasources" \
  -H "Content-Type: application/json" \
  -u "admin:neural_nexus_admin" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://nn-prometheus:9090",
    "access": "proxy",
    "basicAuth": false
  }' && echo "✅ Data Source configurado"

echo "🚀 Grafana listo en http://localhost:3002"
