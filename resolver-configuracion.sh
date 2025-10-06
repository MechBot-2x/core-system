#!/bin/bash

echo "🧠 Resolviendo configuración de Neural Nexus..."
echo "=============================================="

# 1. Limpiar variables de entorno
echo "🗑️  Limpiando variables de entorno..."
unset GITHUB_TOKEN
unset GH_TOKEN
echo "✅ Variables limpiadas"

# 2. Configurar GitHub CLI
echo "🔐 Configurando GitHub CLI..."
gh auth logout 2>/dev/null
gh auth login --web

# 3. Sincronizar con repositorio remoto
echo "🔄 Sincronizando con GitHub..."
git fetch origin

# 4. Intentar hacer pull y resolver conflictos
echo "📥 Actualizando código local..."
if git pull --rebase origin main; then
    echo "✅ Código actualizado exitosamente"
else
    echo "⚠️  Hay conflictos que resolver manualmente"
    echo "   Ejecuta: git status para ver los archivos con conflictos"
    exit 1
fi

# 5. Hacer push de los cambios
echo "📤 Subiendo cambios..."
if git push origin main; then
    echo "🎉 ¡Todo configurado correctamente!"
    echo ""
    echo "📊 Resumen:"
    echo "  ✅ Variables de entorno limpiadas"
    echo "  ✅ GitHub CLI configurado" 
    echo "  ✅ Código sincronizado con GitHub"
    echo "  ✅ Cambios subidos al repositorio"
else
    echo "❌ Error al hacer push"
    exit 1
fi
