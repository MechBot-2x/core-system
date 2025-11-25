#!/bin/bash
echo "🔄 INTEGRANDO LEGADO CHIZHEVSKY EN CORE-SYSTEM"
echo "=============================================="

# Verificar estructura existente
echo "📁 Estructura actual:"
ls -la

# Crear enlaces simbólicos si es necesario
if [ -d "frontend" ]; then
    echo "🔗 Conectando frontend existente con HelioBio..."
    ln -sf ../heliobio frontend/src/heliobio 2>/dev/null || true
fi

if [ -d "src" ]; then
    echo "🔗 Conectando backend existente con HelioBio..."
    ln -sf ../heliobio src/heliobio 2>/dev/null || true
fi

echo "✅ Integración completada"
echo "🌌 Sistema core-system + HelioBio = LEGADO OPERATIVO"
