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
