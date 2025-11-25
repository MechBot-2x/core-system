#!/bin/bash
echo "🔧 CONFIGURACIÓN COMPLETA GRAFANA"
echo "================================"

# Detener Grafana temporalmente
docker stop nn-grafana

# Eliminar base de datos existente (se recreará)
docker exec nn-grafana rm -f /var/lib/grafana/grafana.db

# Reiniciar Grafana
docker start nn-grafana

# Esperar que esté listo
sleep 15

echo "✅ Grafana reiniciado - Contraseña por defecto: admin/admin"
echo "🌐 Acceso: http://localhost:3002"
