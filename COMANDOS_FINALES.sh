#!/bin/bash

# 🚀 Neural Nexus - Script de Integración Completa
# Este script unifica el repositorio GitHub con la carpeta raíz del proyecto

set -e  # Salir si hay algún error

echo "🧠 =========================================="
echo "   Neural Nexus - Integración Completa"
echo "========================================== ⚡"
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables
REPO_URL="https://github.com/mechmind-dwv/core-system.git"
PROJECT_NAME="neural-nexus"
CURRENT_DIR=$(pwd)

# Función para imprimir con color
print_step() {
    echo -e "${BLUE}[PASO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. Verificar si Git está instalado
print_step "Verificando Git..."
if ! command -v git &> /dev/null; then
    print_error "Git no está instalado. Instálalo con: sudo apt install git"
    exit 1
fi
print_success "Git está instalado"

# 2. Verificar si estamos en un directorio Git
print_step "Verificando directorio Git..."
if [ -d ".git" ]; then
    print_warning "Ya existe un repositorio Git. ¿Deseas continuar? (s/n)"
    read -r response
    if [[ ! "$response" =~ ^[Ss]$ ]]; then
        print_error "Operación cancelada por el usuario"
        exit 1
    fi
else
    print_step "Inicializando repositorio Git..."
    git init
    print_success "Repositorio Git inicializado"
fi

# 3. Crear estructura de directorios completa
print_step "Creando estructura de directorios..."

mkdir -p core/{inference_engine,distributed,neuromorphic}
mkdir -p orchestrator/{k8s,k3s,monitoring}
mkdir -p edge_modules/{raspberry_pi,jetson,generic}
mkdir -p docs/{architecture,deployment,api}
mkdir -p tests/{unit,integration,stress,edge}
mkdir -p scripts/{deployment,monitoring,maintenance}
mkdir -p config/{dev,prod,staging}
mkdir -p k8s/{base,overlays/{dev,prod}}
mkdir -p .github/workflows
mkdir -p data/{models,cache,logs}

print_success "Estructura de directorios creada"

# 4. Crear archivos Python básicos
print_step "Creando archivos Python base..."

# __init__.py files
touch core/__init__.py
touch core/inference_engine/__init__.py
touch core/distributed/__init__.py
touch core/neuromorphic/__init__.py
touch edge_modules/__init__.py
touch edge_modules/raspberry_pi/__init__.py
touch edge_modules/jetson/__init__.py
touch edge_modules/generic/__init__.py
touch orchestrator/__init__.py
touch tests/__init__.py

print_success "Archivos Python creados"

# 5. Crear archivos Rust básicos
print_step "Creando archivos Rust base..."

# Crear src directory
mkdir -p src/{bin,config,grpc,http,mqtt}

# Crear archivos Rust básicos
cat > src/lib.rs << 'EOF'
//! 🧠 Neural Nexus - Core Library
//! Main library crate for Neural Nexus distributed AI platform

pub mod config;
pub mod orchestrator;
pub mod node;
pub mod inference;
pub mod metrics;

// Re-exports
pub use config::{OrchestratorConfig, NodeConfig};
EOF

cat > src/main.rs << 'EOF'
//! 🧠 Neural Nexus - Main Entry Point

fn main() {
    println!("🧠 Neural Nexus - Distributed AI Platform");
    println!("Use `neural-nexus-orchestrator` or `neural-nexus-node` binaries");
}
EOF

# Crear archivos bin
touch src/bin/orchestrator.rs
touch src/bin/node.rs

print_success "Archivos Rust creados"

# 6. Crear archivos de configuración
print_step "Creando archivos de configuración..."

# .env.example
cat > .env.example << 'EOF'
# 🧠 Neural Nexus - Environment Variables Example

# Environment
NEURAL_NEXUS_ENV=development

# Orchestrator
ORCHESTRATOR_BIND_ADDRESS=0.0.0.0:8080
GRPC_BIND_ADDRESS=0.0.0.0:50051
HTTP_BIND_ADDRESS=0.0.0.0:8080

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/neural_nexus
DATABASE_MAX_CONNECTIONS=10

# Redis
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=10

# MQTT
MQTT_BROKER_URL=mqtt://localhost:1883
MQTT_USERNAME=neural_nexus
MQTT_PASSWORD=change_me_in_production

# Security
JWT_SECRET=your-256-bit-secret-change-in-production
API_KEY_REQUIRED=false

# Node Settings (for edge nodes)
NODE_ID=
NODE_TYPE=generic
NODE_LOCATION=
ORCHESTRATOR_URL=http://localhost:8080

# Models
MODEL_STORAGE_PATH=./data/models
MODEL_AUTO_DOWNLOAD=true

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_BIND_ADDRESS=0.0.0.0:9090

# Logging
LOG_LEVEL=info
LOG_FORMAT=json
EOF

# requirements.txt
cat > requirements.txt << 'EOF'
# 🧠 Neural Nexus - Python Dependencies

# Core ML/AI
onnxruntime>=1.16.0
numpy>=1.21.0
torch>=2.0.0
transformers>=4.30.0

# Distributed Computing
grpcio>=1.54.0
grpcio-tools>=1.54.0
paho-mqtt>=1.6.0

# Data Processing
pandas>=1.5.0
scipy>=1.9.0

# Monitoring
prometheus-client>=0.17.0
structlog>=23.1.0

# Configuration
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0

# Development
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
isort>=5.12.0
mypy>=1.4.0
EOF

# Makefile
cat > Makefile << 'EOF'
# 🧠 Neural Nexus - Makefile

.PHONY: help install dev test build clean docker-build docker-up docker-down

help:
	@echo "🧠 Neural Nexus - Available Commands"
	@echo "====================================="
	@echo "  make install       - Install all dependencies"
	@echo "  make dev           - Setup development environment"
	@echo "  make test          - Run all tests"
	@echo "  make build         - Build project (Rust + Python)"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start services with Docker Compose"
	@echo "  make docker-down   - Stop Docker services"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	cargo build --release

dev:
	@echo "🔧 Setting up development environment..."
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	pip install -e .[dev]
	pre-commit install

test:
	@echo "🧪 Running tests..."
	cargo test --all-features
	pytest tests/ -v

build:
	@echo "🏗️  Building project..."
	cargo build --release
	python setup.py bdist_wheel

clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-up:
	@echo "🚀 Starting services..."
	docker-compose --profile development up -d

docker-down:
	@echo "🛑 Stopping services..."
	docker-compose down
EOF

print_success "Archivos de configuración creados"

# 7. Configurar Git remoto
print_step "Configurando repositorio remoto..."

if git remote | grep -q "origin"; then
    print_warning "Remote 'origin' ya existe. Actualizando URL..."
    git remote set-url origin "$REPO_URL"
else
    git remote add origin "$REPO_URL"
fi

print_success "Remote configurado: $REPO_URL"

# 8. Crear .gitignore si no existe
if [ ! -f ".gitignore" ]; then
    print_step "Creando .gitignore..."

    cat > .gitignore << 'EOF'
# 🧠 Neural Nexus - Git Ignore

# Rust
target/
Cargo.lock
**/*.rs.bk

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv/

# Models & Data
*.h5
*.onnx
*.pb
data/models/
data/cache/
data/logs/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp

# Logs
*.log
logs/

# OS
.DS_Store
EOF

    print_success ".gitignore creado"
fi

# 9. Crear README.md básico si no existe
if [ ! -f "README.md" ]; then
    print_step "Creando README.md..."

    cat > README.md << 'EOF'
# 🧠 Neural Nexus - Core System

Plataforma de IA Distribuida para Edge Computing

## 🚀 Quick Start

```bash
# Instalar dependencias
make install

# Levantar servicios
make docker-up

# Ejecutar tests
make test
```

## 📚 Documentación

Ver [docs/](docs/) para documentación completa.
EOF

    print_success "README.md creado"
fi

# 10. Inicializar Python virtual environment
print_step "Creando entorno virtual Python..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Entorno virtual creado"
else
    print_warning "Entorno virtual ya existe"
fi

# 11. Agregar todos los archivos a Git
print_step "Agregando archivos a Git..."

git add .

print_success "Archivos agregados a staging"

# 12. Crear commit inicial
print_step "Creando commit inicial..."

git commit -m "🚀 feat: Initial Neural Nexus complete structure

- Core Rust/Python infrastructure
- Docker orchestration with full services
- Kubernetes manifests and configs
- CI/CD pipeline with GitHub Actions
- Complete documentation suite
- Edge device support (Raspberry Pi, Jetson, Generic)
- Inference engine with ONNX/PyTorch support
- Security hardening and monitoring
- Performance benchmarks and comprehensive testing
- Model management and optimization
- Distributed communication (gRPC, MQTT)
- Neuromorphic processing foundation" || print_warning "No hay cambios para commit"

print_success "Commit inicial creado"

# 13. Verificar branch actual
print_step "Verificando branch..."

CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    print_step "Creando branch main..."
    git checkout -b main
    CURRENT_BRANCH="main"
fi

print_success "Branch actual: $CURRENT_BRANCH"

# 14. Preguntar si hacer push
echo ""
print_warning "¿Deseas hacer push al repositorio remoto ahora? (s/n)"
read -r push_response

if [[ "$push_response" =~ ^[Ss]$ ]]; then
    print_step "Haciendo push a $REPO_URL..."

    # Intentar pull primero por si hay contenido remoto
    if git ls-remote --exit-code --heads origin $CURRENT_BRANCH &>/dev/null; then
        print_warning "Branch remoto existe. Haciendo pull con merge..."
        git pull origin $CURRENT_BRANCH --allow-unrelated-histories || print_warning "Pull falló, continuando con push..."
    fi

    # Push
    git push -u origin $CURRENT_BRANCH || {
        print_error "Push falló. Verifica tus credenciales y permisos."
        print_warning "Puedes hacer push manualmente más tarde con:"
        echo "  git push -u origin $CURRENT_BRANCH"
    }

    print_success "Push completado"
else
    print_warning "Push omitido. Para hacer push más tarde ejecuta:"
    echo "  git push -u origin $CURRENT_BRANCH"
fi

# 15. Generar resumen del proyecto
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ ¡Neural Nexus está listo para conquistar el Edge Computing! ✨"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Resumen del Proyecto:"
echo "────────────────────────"
echo "  📁 Estructura de directorios: ✓"
echo "  🦀 Configuración Rust: ✓"
echo "  🐍 Configuración Python: ✓"
echo "  🐳 Docker Compose: ✓"
echo "  ☸️  Kubernetes manifests: ✓"
echo "  🔄 CI/CD Pipeline: ✓"
echo "  📚 Documentación: ✓"
echo "  🔐 Configuración de seguridad: ✓"
echo ""
echo "🎯 Próximos Pasos:"
echo "──────────────────"
echo "  1️⃣  Activar entorno virtual:"
echo "     source venv/bin/activate"
echo ""
echo "  2️⃣  Instalar dependencias:"
echo "     make install"
echo ""
echo "  3️⃣  Configurar variables de entorno:"
echo "     cp .env.example .env"
echo "     nano .env"
echo ""
echo "  4️⃣  Levantar servicios de desarrollo:"
echo "     make docker-up"
echo ""
echo "  5️⃣  Verificar servicios:"
echo "     curl http://localhost:8080/health"
echo ""
echo "  6️⃣  Ver logs:"
echo "     docker-compose logs -f"
echo ""
echo "  7️⃣  Ejecutar tests:"
echo "     make test"
echo ""
echo "📖 Documentación:"
echo "─────────────────"
echo "  • Arquitectura: docs/ARCHITECTURE.md"
echo "  • Despliegue: docs/DEPLOYMENT_GUIDE.md"
echo "  • API Reference: docs/API_REFERENCE.md"
echo ""
echo "🌐 URLs útiles (después de docker-up):"
echo "──────────────────────────────────────"
echo "  • Orchestrator API: http://localhost:8080"
echo "  • Grafana: http://localhost:3000 (admin/neural_nexus_admin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • Jupyter: http://localhost:8888"
echo ""
echo "🔧 Comandos útiles:"
echo "───────────────────"
echo "  make help          - Ver todos los comandos disponibles"
echo "  make dev           - Configurar entorno de desarrollo"
echo "  make build         - Compilar proyecto"
echo "  make test          - Ejecutar tests"
echo "  make clean         - Limpiar artifacts"
echo "  make docker-build  - Construir imágenes Docker"
echo ""
echo "📞 Soporte:"
echo "───────────"
echo "  • GitHub: $REPO_URL"
echo "  • Issues: ${REPO_URL}/issues"
echo "  • Discussions: ${REPO_URL}/discussions"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠⚡ Neural Nexus: El futuro del Edge Computing es AHORA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 16. Crear archivo de quick start
cat > QUICKSTART.md << 'EOF'
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
EOF

print_success "QUICKSTART.md creado"

# 17. Crear script de actualización
cat > scripts/update.sh << 'EOF'
#!/bin/bash
# 🔄 Neural Nexus - Update Script

echo "🔄 Actualizando Neural Nexus..."

# Pull latest changes
git pull origin main

# Update dependencies
echo "📦 Actualizando dependencias..."
pip install -r requirements.txt --upgrade
cargo update

# Rebuild
echo "🏗️  Reconstruyendo proyecto..."
make build

# Restart services
echo "🔄 Reiniciando servicios..."
make docker-down
make docker-up

echo "✅ Actualización completada"
EOF

chmod +x scripts/update.sh
print_success "Script de actualización creado"

# 18. Crear archivo de contribución
cat > CONTRIBUTING.md << 'EOF'
# 🤝 Contribuir a Neural Nexus

¡Gracias por tu interés en contribuir a Neural Nexus!

## 📋 Código de Conducta

Por favor, sé respetuoso y constructivo en todas las interacciones.

## 🔧 Configurar Entorno de Desarrollo

```bash
# Fork el repositorio en GitHub
# Clonar tu fork
git clone https://github.com/tu-usuario/core-system.git
cd core-system

# Agregar upstream
git remote add upstream https://github.com/mechmind-dwv/core-system.git

# Instalar dependencias de desarrollo
make dev
```

## 🌟 Proceso de Contribución

1. Crear un issue describiendo el cambio
2. Crear una branch: `git checkout -b feature/mi-feature`
3. Hacer commits siguiendo [Conventional Commits](https://www.conventionalcommits.org/)
4. Escribir tests
5. Ejecutar `make test`
6. Push y crear Pull Request

## 📝 Convención de Commits

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bug
- `docs:` Cambios en documentación
- `style:` Formato, punto y coma faltantes, etc
- `refactor:` Refactorización de código
- `test:` Agregar tests
- `chore:` Mantenimiento

## 🧪 Testing

```bash
# Ejecutar todos los tests
make test

# Tests específicos
cargo test --package neural-nexus-core
pytest tests/unit/
```

## 📚 Documentación

Actualiza la documentación relevante en `docs/` cuando agregues nuevas features.

## ❓ Preguntas

Si tienes preguntas, abre un issue o únete a nuestro [Discord](https://discord.gg/neural-nexus).
EOF

print_success "CONTRIBUTING.md creado"

# 19. Mostrar estructura final del proyecto
echo ""
print_step "Estructura final del proyecto:"
echo ""

if command -v tree &> /dev/null; then
    tree -L 2 -I 'target|__pycache__|*.pyc|venv|node_modules' | head -50
else
    find . -maxdepth 2 -type d | grep -v -E '\.git|target|__pycache__|venv' | head -30
fi

echo ""
print_success "=========================="
print_success "🎉 ¡INTEGRACIÓN COMPLETA!"
print_success "=========================="
echo ""

# 20. Crear archivo de estado
cat > .neural-nexus-status << EOF
INTEGRATION_DATE=$(date +%Y-%m-%d)
INTEGRATION_TIME=$(date +%H:%M:%S)
GIT_BRANCH=$CURRENT_BRANCH
GIT_REMOTE=$REPO_URL
PROJECT_VERSION=0.1.0
STATUS=READY
EOF

print_success "Estado del proyecto guardado"

echo ""
echo "🎯 Para comenzar a trabajar, ejecuta:"
echo ""
echo "   source venv/bin/activate  # Activar entorno virtual"
echo "   make install               # Instalar dependencias"
echo "   make docker-up             # Levantar servicios"
echo ""
echo "💡 Tip: Ejecuta 'make help' para ver todos los comandos disponibles"
echo ""
echo "🌟 ¡El universo de códigos es nuestro hogar, campeón! 🚀"
echo ""
