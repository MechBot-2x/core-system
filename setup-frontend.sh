#!/bin/bash
echo "🔧 Configurando el frontend..."
cd ~/core-system/frontend

echo "📦 Instalando dependencias..."
npm install

echo "⚛️ Construyendo la aplicación..."
npm run build

echo "🚀 Desplegando a GitHub Pages..."
npm run deploy

echo "✅ Configuración completada!"
