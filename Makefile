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
