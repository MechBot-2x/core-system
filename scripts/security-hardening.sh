#!/bin/bash
# scripts/security-hardening.sh

echo "🔐 Aplicando configuraciones de seguridad..."

# Configurar firewall
sudo ufw --force enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Puertos necesarios
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8080/tcp  # Neural Nexus API
sudo ufw allow 50051/tcp # gRPC

# Configurar fail2ban
sudo apt-get install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Configurar auditd
sudo apt-get install -y auditd
sudo systemctl enable auditd
