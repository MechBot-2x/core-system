#!/bin/bash
# scripts/cleanup.sh

echo "🧹 Iniciando limpieza del sistema..."

# Limpiar logs antiguos (>7 días)
find /opt/neural-nexus/logs -name "*.log" -mtime +7 -delete

# Limpiar modelos no utilizados
docker exec neural-nexus-orchestrator \
  /app/bin/cleanup-unused-models --older-than=30d

# Optimizar base de datos
docker exec neural-nexus-postgres \
  psql -U postgres -d neural_nexus -c "VACUUM ANALYZE;"

# Limpiar imágenes Docker no utilizadas
docker system prune -f

echo "✅ Limpieza completada"
