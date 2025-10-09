# 🚀 Neural Nexus - Quick Start Guide

## Instalación Rápida

### 1. Clonar Repositorio
```bash
git clone https://github.com/mechmind-dwv/core-system.git
cd core-system
```

### 2. Configuración Inicial
```bash
# Copiar configuración de ejemplo
cp .env.example .env

# Editar variables de entorno
nano .env
```

### 3. Desarrollo Local

#### Opción A: Docker (Recomendado)
```bash
# Levantar todos los servicios
make docker-up

# Verificar servicios
docker-compose ps

# Ver logs
docker-compose logs -f neural-nexus-orchestrator
```

#### Opción B: Desarrollo Nativo
```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
make install

# Ejecutar orchestrator
cargo run --bin neural-nexus-orchestrator

# En otra terminal: ejecutar node
cargo run --bin neural-nexus-node
```

### 4. Verificar Instalación
```bash
# Health check del orchestrator
curl http://localhost:8080/health

# Métricas
curl http://localhost:8080/metrics

# Lista de nodos
curl http://localhost:8080/api/v1/nodes
```

### 5. Despliegue en Edge Device

#### Raspberry Pi
```bash
# SSH a tu Raspberry Pi
ssh pi@raspberrypi.local

# Ejecutar instalador
curl -sSL https://install.neural-nexus.dev/rpi | bash

# O manualmente con Docker
docker run -d --name neural-nexus-rpi \
  --restart unless-stopped \
  -v /opt/neural-nexus:/app/data \
  -p 8080:8080 \
  -e ORCHESTRATOR_URL=http://your-orchestrator:8080 \
  neuralnexus/raspberry-pi:latest
```

#### NVIDIA Jetson
```bash
# Con GPU acceleration
docker run -d --name neural-nexus-jetson \
  --runtime nvidia \
  --restart unless-stopped \
  -v /opt/neural-nexus:/app/data \
  -p 8080:8080 \
  -e GPU_ACCELERATION=true \
  neuralnexus/jetson:latest
```

## Primeros Pasos

### Registrar un Nodo Edge
```python
import requests

response = requests.post('http://localhost:8080/api/v1/nodes/register', json={
    'node_id': 'my-edge-node-001',
    'node_type': 'raspberry-pi',
    'capabilities': ['inference', 'monitoring'],
    'hardware_info': {
        'cpu_cores': 4,
        'memory_mb': 4096,
        'gpu_available': False
    }
})

print(response.json())
```

### Ejecutar Inferencia
```python
import requests
import numpy as np

# Preparar datos
input_data = np.random.randn(1, 3, 224, 224).tolist()

# Enviar request
response = requests.post('http://localhost:8080/api/v1/inference', json={
    'model_name': 'resnet50',
    'input_data': input_data,
    'priority': 5
})

result = response.json()
print(f"Inferencia completada en {result['processing_time_ms']}ms")
print(f"Outputs: {result['outputs']}")
```

### Monitorear Métricas
```bash
# Acceder a Grafana
open http://localhost:3000

# Usuario: admin
# Contraseña: neural_nexus_admin

# Dashboards pre-configurados:
# - Sistema Overview
# - Node Performance
# - Inference Metrics
# - Resource Usage
```

## Troubleshooting

### Problema: Servicios no arrancan
```bash
# Verificar logs
docker-compose logs

# Reiniciar servicios
make docker-down
make docker-up

# Verificar puertos
sudo netstat -tulpn | grep -E '8080|50051|5432|6379|1883'
```

### Problema: Nodo no se conecta al orchestrator
```bash
# Verificar conectividad
ping orchestrator-hostname

# Verificar variables de entorno
echo $ORCHESTRATOR_URL

# Ver logs del nodo
docker logs neural-nexus-node
```

### Problema: Alta latencia en inferencia
```bash
# Verificar uso de recursos
docker stats

# Optimizar configuración
# Editar config/node.toml
[inference.optimization]
quantization.enabled = true
batching.max_batch_size = 16
```

## Recursos Adicionales

- 📖 [Documentación Completa](docs/)
- 🏗️ [Arquitectura del Sistema](docs/ARCHITECTURE.md)
- 🚀 [Guía de Despliegue](docs/DEPLOYMENT_GUIDE.md)
- 🔌 [API Reference](docs/API_REFERENCE.md)
- 🤝 [Guía de Contribución](CONTRIBUTING.md)
- 💬 [Discord Community](https://discord.gg/neural-nexus)

## Siguientes Pasos

1. ✅ Configurar tu primer nodo edge
2. ✅ Cargar tu primer modelo ML
3. ✅ Ejecutar inferencias
4. ✅ Monitorear métricas en Grafana
5. ✅ Escalar horizontalmente con Kubernetes
6. ✅ Implementar aprendizaje federado
7. ✅ Contribuir al proyecto

---

> 🧠⚡ **¡Bienvenido a Neural Nexus!** El futuro del Edge Computing distribuido está en tus manos.
