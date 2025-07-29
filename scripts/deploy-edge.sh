# Script de auto-configuración
#!/bin/bash
# scripts/deploy-edge.sh

DEVICE_TYPE=${1:-generic}
ORCHESTRATOR_URL=${2:-http://localhost:8080}

echo "🚀 Desplegando Neural Nexus en $DEVICE_TYPE"

# Detectar arquitectura
ARCH=$(uname -m)
case $ARCH in
    armv7l) IMAGE_TAG="arm32v7" ;;
    aarch64) IMAGE_TAG="arm64v8" ;;
    x86_64) IMAGE_TAG="amd64" ;;
    *) echo "Arquitectura no soportada: $ARCH"; exit 1 ;;
esac

# Crear directorio de configuración
sudo mkdir -p /opt/neural-nexus/{config,data,models,logs}

# Generar configuración
cat > /opt/neural-nexus/config/node.toml << EOF
[node]
id = "$(hostname)-$(date +%s)"
type = "$DEVICE_TYPE"
location = "$(hostname)"

[orchestrator]
url = "$ORCHESTRATOR_URL"
heartbeat_interval = 30

[inference]
model_path = "/app/models/current.onnx"
batch_size = 1
max_latency_ms = 100
EOF

# Desplegar contenedor
docker run -d \
  --name neural-nexus-$DEVICE_TYPE \
  --restart unless-stopped \
  -v /opt/neural-nexus:/app \
  -p 8080:8080 \
  neuralnexus/edge-$IMAGE_TAG:latest

echo "✅ Neural Nexus desplegado exitosamente"
