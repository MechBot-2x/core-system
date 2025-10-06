#!/bin/bash
# scripts/integration-tests.sh

echo "🧪 Ejecutando tests de integración..."

# Test de conectividad
curl -f http://localhost:8080/health || exit 1
curl -f http://localhost:8081/health || exit 1

# Test de inferencia
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "data": [1,2,3,4]}' || exit 1

# Test de métricas
curl -f http://localhost:8080/metrics | grep neural_nexus || exit 1

# Test de comunicación MQTT
python3 tests/test_mqtt_communication.py || exit 1

echo "✅ Todos los tests pasaron"
