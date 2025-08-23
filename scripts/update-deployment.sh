#!/bin/bash
# scripts/update-deployment.sh

set -e

echo "🔄 Iniciando actualización de Neural Nexus..."

# Backup de configuración
kubectl create backup neural-nexus-config-$(date +%Y%m%d-%H%M%S) \
  --from-file=/opt/neural-nexus/config

# Rolling update del orchestrator
kubectl set image deployment/neural-nexus-orchestrator \
  orchestrator=neuralnexus/orchestrator:latest \
  -n neural-nexus

# Esperar a que el rollout complete
kubectl rollout status deployment/neural-nexus-orchestrator -n neural-nexus

# Actualizar nodos edge
kubectl set image daemonset/neural-nexus-edge \
  edge-node=neuralnexus/edge-node:latest \
  -n neural-nexus

# Verificar health checks
sleep 30
kubectl get pods -n neural-nexus
kubectl exec -n neural-nexus deployment/neural-nexus-orchestrator -- \
  curl -f http://localhost:8080/health

echo "✅ Actualización completada exitosamente"
