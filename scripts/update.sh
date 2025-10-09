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
