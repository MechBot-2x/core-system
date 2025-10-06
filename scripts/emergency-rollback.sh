#!/bin/bash
# scripts/emergency-rollback.sh

echo "🚨 Iniciando rollback de emergencia..."

# Rollback del orchestrator
kubectl rollout undo deployment/neural-nexus-orchestrator -n neural-nexus

# Rollback de nodos edge
kubectl rollout undo daemonset/neural-nexus-edge -n neural-nexus

# Verificar estado
kubectl get pods -n neural-nexus
kubectl logs -f deployment/neural-nexus-orchestrator -n neural-nexus

echo "🔙 Rollback completado"
