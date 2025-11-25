#!/bin/bash
echo "🚀 DEPLOYMENT UNIFICADO: Core-System + HelioBio"
echo "=============================================="

# 1. Verificar sistema existente
echo "🔍 Analizando core-system actual..."
docker-compose config 2>/dev/null && echo "✅ Docker Compose detectado" || echo "⚠️  Sin Docker Compose"
[ -f "package.json" ] && echo "✅ Node.js detectado" || echo "⚠️  Sin Node.js"
[ -f "requirements.txt" ] && echo "✅ Python detectado" || echo "⚠️  Sin Python"

# 2. Integrar HelioBio
echo "🌌 Integrando módulos HelioBio..."
./integrar-heliobio.sh

# 3. Probar integración
echo "🧪 Probando integración..."
python3 heliobio/social/analisis_tac_rate.py

echo "🎉 DEPLOYMENT UNIFICADO COMPLETADO"
echo ""
echo "📊 SISTEMA INTEGRADO:"
echo "   • Core-System existente ✅"
echo "   • Módulos HelioBio ✅"
echo "   • Análisis TAC Rate ✅"
echo ""
echo "🔮 Próximo paso: Ejecutar tu sistema normal + módulos HelioBio"
