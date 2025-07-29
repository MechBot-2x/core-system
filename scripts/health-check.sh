#!/bin/bash
# scripts/health-check.sh

echo "🏥 Neural Nexus Health Check"
echo "=========================="

# 1. Servicios básicos
echo "1. Verificando servicios..."
systemctl is-active docker || echo "❌ Docker no está activo"
systemctl is-active kubelet || echo "❌ Kubelet no está activo"

# 2. Conectividad de red
echo "2. Verificando conectividad..."
ping -c 1 8.8.8.8 > /dev/null || echo "❌ Sin conexión a internet"
nc -zv localhost 8080 || echo "❌ Orchestrator no responde"

# 3. Recursos del sistema
echo "3. Verificando recursos..."
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

echo "Uso de memoria: ${MEMORY_USAGE}%"
echo "Uso de CPU: ${CPU_USAGE}%"

# 4. Espacio en disco
echo "4. Verificando espacio en disco..."
df -h | grep -E "(/$|/opt)" | while read line; do
    USAGE=$(echo $line | awk '{print $5}' | cut -d'%' -f1)
    if [ $USAGE -gt 80 ]; then
        echo "❌ Poco espacio en disco: $line"
    fi
done

# 5. Logs de errores recientes
echo "5. Verificando logs recientes..."
ERROR_COUNT=$(journalctl -u neural-nexus --since "1 hour ago" | grep -i error | wc -l)
if [ $ERROR_COUNT -gt 0 ]; then
    echo "⚠️  $ERROR_COUNT errores en la última hora"
fi

echo "✅ Health check completado"
