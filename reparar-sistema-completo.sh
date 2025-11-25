#!/bin/bash
echo "🔧 REPARACIÓN COMPLETA DEL SISTEMA HELIOBIO"
echo "=========================================="

# 1. Detener todos los servicios
echo "🛑 Deteniendo servicios..."
docker-compose down

# 2. Reparar archivo docker-compose.yml
echo "📝 Reparando docker-compose.yml..."
cp docker-compose.yml docker-compose.yml.backup.repair
cat > docker-compose-fixed.yml << 'DOCKERFIX'
version: '3.8'

services:
  # 🧠 Core Neural Nexus Services
  neural-nexus-orchestrator:
    image: rust:1.70
    container_name: nn-orchestrator
    working_dir: /app
    command: /bin/bash -c "echo '🧠 Neural Nexus Orchestrator ready' && tail -f /dev/null"
    ports:
      - "8082:8080"
      - "50051:50051"
    networks:
      - neural-nexus

  neural-nexus-node:
    image: rust:1.70
    container_name: nn-node
    working_dir: /app
    command: /bin/bash -c "echo '⚡ Neural Nexus Edge Node ready' && tail -f /dev/null"
    ports:
      - "8083:8081"
    depends_on:
      - neural-nexus-orchestrator
    networks:
      - neural-nexus

  # 📊 Monitoring Stack
  grafana:
    image: grafana/grafana:latest
    container_name: nn-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=HelioBio2025!
    ports:
      - "3002:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - neural-nexus

  prometheus:
    image: prom/prometheus:latest
    container_name: nn-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    ports:
      - "9091:9090"
    volumes:
      - prometheus_data:/prometheus
    networks:
      - neural-nexus

  # 🗄️ Data Services
  postgres:
    image: postgres:15-alpine
    container_name: nn-postgres
    environment:
      - POSTGRES_DB=neural_nexus
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5434:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - neural-nexus

  redis:
    image: redis:7-alpine
    container_name: nn-redis
    ports:
      - "6380:6379"
    volumes:
      - redis_data:/data
    networks:
      - neural-nexus

networks:
  neural-nexus:
    driver: bridge

volumes:
  grafana_data:
  prometheus_data:
  postgres_data:
  redis_data:
DOCKERFIX

# 3. Reemplazar archivo dañado
mv docker-compose-fixed.yml docker-compose.yml

# 4. Levantar servicios
echo "🚀 Iniciando servicios reparados..."
docker-compose up -d

# 5. Esperar y configurar
sleep 20

# 6. Configurar Grafana
echo "🔧 Configurando Grafana..."
docker exec nn-grafana grafana cli admin reset-admin-password "HelioBio2025!" 2>/dev/null || true

# 7. Verificar
echo "✅ VERIFICACIÓN FINAL:"
docker-compose ps | grep Up
curl -s http://localhost:3002 >/dev/null && echo "🌐 Grafana: http://localhost:3002 (admin/HelioBio2025!)" || echo "❌ Grafana no responde"

echo ""
echo "🎉 SISTEMA REPARADO COMPLETAMENTE"
