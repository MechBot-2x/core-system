#!/bin/bash

# Neural Nexus - Automatic Token and Permissions Setup Script
# This script automates GitHub token configuration and permission setup

echo "🧠 Neural Nexus - Token & Permissions Auto-Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log_info() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️ $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Check if running with sudo privileges
check_privileges() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root. Please run as regular user."
        exit 1
    fi
}

# Setup GitHub token configuration
setup_github_token() {
    log_info "Setting up GitHub token configuration..."
    
    # Check if token is already set in git config
    if git config --global github.token &>/dev/null; then
        log_warn "GitHub token already configured in git config"
        return 0
    fi
    
    # Prompt for token
    echo
    log_warn "Please enter your GitHub Personal Access Token"
    log_warn "You can create one at: https://github.com/settings/tokens"
    echo
    read -sp "Enter GitHub Token: " GITHUB_TOKEN
    echo
    
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "No token provided. Skipping token setup."
        return 1
    fi
    
    # Configure token in git
    git config --global github.token "$GITHUB_TOKEN"
    
    # Setup credential helper to use token
    git config --global credential.helper store
    
    # Add token to remote URL if repository exists
    if git remote get-url origin &>/dev/null; then
        CURRENT_URL=$(git remote get-url origin)
        if [[ $CURRENT_URL != *"$GITHUB_TOKEN"* ]]; then
            NEW_URL=$(echo "$CURRENT_URL" | sed "s|https://|https://$GITHUB_TOKEN@|")
            git remote set-url origin "$NEW_URL"
            log_info "Updated remote URL with token"
        fi
    fi
    
    log_info "GitHub token configured successfully"
}

# Setup project permissions
setup_project_permissions() {
    log_info "Setting up project permissions..."
    
    # Make all scripts executable
    find ~/core-system/scripts -name "*.sh" -type f -exec chmod +x {} \;
    log_info "Made all scripts executable"
    
    # Setup neural-nexus directories with correct permissions
    sudo mkdir -p /opt/neural-nexus/{config,data,models,logs}
    sudo chown -R $USER:$USER /opt/neural-nexus/
    sudo chmod 755 /opt/neural-nexus/
    sudo chmod -R 755 /opt/neural-nexus/{config,data,models,logs}
    
    log_info "Neural Nexus directories created with correct permissions"
    
    # Fix permissions for key project files
    chmod +x ~/core-system/get-docker.sh 2>/dev/null || true
    chmod +x ~/core-system/setup-frontend.sh 2>/dev/null || true
    
    # Setup git pre-commit to allow missing config
    git config --global --unset-all core.hooksPath 2>/dev/null || true
    pre-commit install --allow-missing-config 2>/dev/null || true
}

# Setup environment variables
setup_environment() {
    log_info "Setting up environment variables..."
    
    # Add to bashrc if not already present
    if ! grep -q "NEURAL_NEXUS_HOME" ~/.bashrc; then
        echo "export NEURAL_NEXUS_HOME=\$HOME/core-system" >> ~/.bashrc
        echo "export PATH=\$PATH:\$HOME/core-system/scripts" >> ~/.bashrc
    fi
    
    # Source bashrc to apply changes
    source ~/.bashrc
    
    log_info "Environment variables configured"
}

# Security hardening for tokens
setup_token_security() {
    log_info "Setting up token security..."
    
    # Create secure token directory
    mkdir -p ~/.config/neural-nexus
    chmod 700 ~/.config/neural-nexus
    
    # Setup token cleanup on logout (basic security)
    if ! grep -q "token_cleanup" ~/.bash_logout 2>/dev/null; then
        echo "# Neural Nexus token cleanup" >> ~/.bash_logout
        echo "rm -f ~/.config/neural-nexus/token_*.tmp 2>/dev/null" >> ~/.bash_logout
    fi
    
    log_warn "For production use, consider using:"
    log_warn "1. GitHub CLI (gh) for authentication"
    log_warn "2. System keyring for token storage"
    log_warn "3. Environment variables in CI/CD"
}

# Verify setup
verify_setup() {
    log_info "Verifying setup..."
    
    echo
    echo "📊 Setup Verification:"
    echo "-------------------"
    
    # Check git token
    if git config --global github.token &>/dev/null; then
        echo -e "${GREEN}✅ GitHub token configured${NC}"
    else
        echo -e "${YELLOW}⚠️ GitHub token not configured${NC}"
    fi
    
    # Check script permissions
    SCRIPTS_EXECUTABLE=$(find ~/core-system/scripts -name "*.sh" -type f ! -executable | wc -l)
    if [[ $SCRIPTS_EXECUTABLE -eq 0 ]]; then
        echo -e "${GREEN}✅ All scripts are executable${NC}"
    else
        echo -e "${YELLOW}⚠️ $SCRIPTS_EXECUTABLE scripts need execute permission${NC}"
    fi
    
    # Check neural-nexus directories
    if [[ -d "/opt/neural-nexus" ]]; then
        echo -e "${GREEN}✅ Neural Nexus directories created${NC}"
    else
        echo -e "${RED}❌ Neural Nexus directories missing${NC}"
    fi
    
    # Check kubectl connectivity
    if kubectl cluster-info &>/dev/null; then
        echo -e "${GREEN}✅ Kubernetes cluster accessible${NC}"
    else
        echo -e "${YELLOW}⚠️ Kubernetes cluster not accessible${NC}"
    fi
}

# Main execution
main() {
    echo "Starting automatic configuration..."
    echo
    
    check_privileges
    setup_github_token
    setup_project_permissions
    setup_environment
    setup_token_security
    verify_setup
    
    echo
    log_info "Automatic configuration completed!"
    echo
    log_warn "Next steps:"
    echo "1. Run: git push to test token configuration"
    echo "2. Run: ./scripts/health-check.sh to verify system"
    echo "3. Run: docker-compose up -d to start services"
    echo "4. Configure your IDE with project settings"
}

# Run main function
main "$@"
