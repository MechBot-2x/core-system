// 🧠 Neural Nexus - Core Orchestrator
// Main orchestrator binary for managing edge nodes

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use uuid::Uuid;

// Core modules
mod config;
mod grpc_server;
mod http_server;
mod node_manager;
mod metrics;
mod database;

use config::OrchestratorConfig;
use node_manager::{NodeManager, NodeInfo, NodeStatus};
use metrics::MetricsCollector;

/// Main orchestrator state
#[derive(Clone)]
pub struct OrchestratorState {
    pub node_manager: Arc<NodeManager>,
    pub metrics: Arc<MetricsCollector>,
    pub config: Arc<OrchestratorConfig>,
    pub db: Arc<database::Database>,
}

/// Node registration request
#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterNodeRequest {
    pub node_id: String,
    pub node_type: String,
    pub capabilities: Vec<String>,
    pub hardware_info: HardwareInfo,
    pub location: Option<String>,
}

/// Hardware information
#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub gpu_available: bool,
    pub gpu_memory_mb: Option<u64>,
    pub architecture: String, // x86, arm64, etc.
    pub power_consumption_watts: Option<f64>,
}

/// Inference request routing
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub model_name: String,
    pub input_data: Vec<f32>,
    pub priority: InferencePriority,
    pub max_latency_ms: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum InferencePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Main orchestrator logic
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("neural_nexus=info,tower_http=debug")
        .init();

    info!("🧠 Starting Neural Nexus Core Orchestrator");

    // Load configuration
    let config = Arc::new(
        OrchestratorConfig::from_env()
            .context("Failed to load orchestrator configuration")?
    );

    info!("📋 Configuration loaded: {:?}", config.orchestrator.bind_address);

    // Initialize database connection
    let db = Arc::new(
        database::Database::connect(&config.database.url).await
            .context("Failed to connect to database")?
    );

    info!("💾 Database connected successfully");

    // Initialize metrics collector
    let metrics = Arc::new(MetricsCollector::new());
    
    // Initialize node manager
    let node_manager = Arc::new(NodeManager::new(
        config.clone(),
        db.clone(),
        metrics.clone(),
    ));

    // Create shared state
    let state = OrchestratorState {
        node_manager: node_manager.clone(),
        metrics: metrics.clone(),
        config: config.clone(),
        db: db.clone(),
    };

    info!("🎛️ Core components initialized");

    // Start background tasks
    let heartbeat_task = tokio::spawn(heartbeat_monitor(node_manager.clone()));
    let metrics_task = tokio::spawn(metrics_collector_task(metrics.clone()));
    let load_balancer_task = tokio::spawn(load_balancer_optimizer(state.clone()));

    info!("🔄 Background tasks started");

    // Start servers
    let grpc_server = tokio::spawn(grpc_server::start_grpc_server(
        state.clone(),
        config.grpc.bind_address.clone(),
    ));

    let http_server = tokio::spawn(http_server::start_http_server(
        state.clone(),
        config.http.bind_address.clone(),
    ));

    info!("🚀 Neural Nexus Orchestrator started successfully!");
    info!("📡 gRPC server: {}", config.grpc.bind_address);
    info!("🌐 HTTP server: {}", config.http.bind_address);

    // Wait for servers or handle shutdown
    tokio::select! {
        result = grpc_server => {
            error!("gRPC server exited: {:?}", result);
        },
        result = http_server => {
            error!("HTTP server exited: {:?}", result);
        },
        _ = tokio::signal::ctrl_c() => {
            info!("🛑 Received shutdown signal");
        }
    }

    // Graceful shutdown
    info!("🔄 Initiating graceful shutdown...");
    
    // Cancel background tasks
    heartbeat_task.abort();
    metrics_task.abort();
    load_balancer_task.abort();

    // Save final state
    if let Err(e) = save_orchestrator_state(&state).await {
        warn!("Failed to save final state: {}", e);
    }

    info!("✅ Neural Nexus Orchestrator shutdown complete");
    Ok(())
}

/// Background task to monitor node heartbeats
async fn heartbeat_monitor(node_manager: Arc<NodeManager>) {
    let mut interval = tokio::time::interval(
        std::time::Duration::from_secs(30)
    );

    loop {
        interval.tick().await;
        
        if let Err(e) = node_manager.check_heartbeats().await {
            error!("Heartbeat check failed: {}", e);
        }
    }
}

/// Background task to collect and aggregate metrics
async fn metrics_collector_task(metrics: Arc<MetricsCollector>) {
    let mut interval = tokio::time::interval(
        std::time::Duration::from_secs(10)
    );

    loop {
        interval.tick().await;
        
        metrics.collect_system_metrics().await;
        
        if let Err(e) = metrics.flush_to_prometheus().await {
            warn!("Failed to flush metrics: {}", e);
        }
    }
}

/// Background task to optimize load balancing
async fn load_balancer_optimizer(state: OrchestratorState) {
    let mut interval = tokio::time::interval(
        std::time::Duration::from_secs(60)
    );

    loop {
        interval.tick().await;
        
        if let Err(e) = optimize_node_assignments(&state).await {
            warn!("Load balancer optimization failed: {}", e);
        }
    }
}

/// Optimize node assignments based on current load and capabilities
async fn optimize_node_assignments(state: &OrchestratorState) -> Result<()> {
    let nodes = state.node_manager.get_all_nodes().await?;
    let metrics = state.metrics.get_node_metrics().await?;
    
    // Simple load balancing algorithm
    for (node_id, node_info) in nodes {
        if let Some(node_metrics) = metrics.get(&node_id) {
            // If node is overloaded (>80% CPU), redistribute load
            if node_metrics.cpu_usage > 0.8 {
                info!("🔄 Node {} overloaded, redistributing", node_id);
                
                if let Err(e) = state.node_manager
                    .redistribute_load(&node_id).await {
                    warn!("Failed to redistribute load for {}: {}", node_id, e);
                }
            }
            
            // If node is underutilized (<20% CPU), consider adding load
            if node_metrics.cpu_usage < 0.2 && node_info.status == NodeStatus::Healthy {
                if let Err(e) = state.node_manager
                    .increase_load(&node_id).await {
                    warn!("Failed to increase load for {}: {}", node_id, e);
                }
            }
        }
    }
    
    Ok(())
}

/// Save orchestrator state for recovery
async fn save_orchestrator_state(state: &OrchestratorState) -> Result<()> {
    let nodes = state.node_manager.get_all_nodes().await?;
    let metrics = state.metrics.export_current_state().await?;
    
    // Save to database
    state.db.save_orchestrator_snapshot(nodes, metrics).await?;
    
    info!("💾 Orchestrator state saved successfully");
    Ok(())
}

/// Node registration handler
pub async fn register_node(
    state: &OrchestratorState,
    request: RegisterNodeRequest,
) -> Result<String> {
    info!("📝 Registering new node: {}", request.node_id);

    // Validate node capabilities
    validate_node_capabilities(&request)?;

    // Create node info
    let node_info = NodeInfo {
        id: request.node_id.clone(),
        node_type: request.node_type,
        status: NodeStatus::Initializing,
        capabilities: request.capabilities,
        hardware_info: request.hardware_info,
        location: request.location,
        last_heartbeat: chrono::Utc::now(),
        load_score: 0.0,
        assigned_models: Vec::new(),
    };

    // Register with node manager
    let registration_token = state.node_manager
        .register_node(node_info).await?;

    // Update metrics
    state.metrics.increment_registered_nodes().await;

    info!("✅ Node {} registered successfully", request.node_id);
    Ok(registration_token)
}

/// Validate node capabilities and requirements
fn validate_node_capabilities(request: &RegisterNodeRequest) -> Result<()> {
    // Check minimum requirements
    if request.hardware_info.cpu_cores < 1 {
        anyhow::bail!("Node must have at least 1 CPU core");
    }
    
    if request.hardware_info.memory_mb < 512 {
        anyhow::bail!("Node must have at least 512MB of memory");
    }

    // Validate node type
    match request.node_type.as_str() {
        "raspberry-pi" | "jetson" | "generic" | "server" => {},
        _ => anyhow::bail!("Unsupported node type: {}", request.node_type),
    }

    // Validate capabilities
    for capability in &request.capabilities {
        match capability.as_str() {
            "inference" | "training" | "preprocessing" | "monitoring" => {},
            _ => anyhow::bail!("Unknown capability: {}", capability),
        }
    }

    Ok(())
}

/// Route inference request to optimal node
pub async fn route_inference_request(
    state: &OrchestratorState,
    request: InferenceRequest,
) -> Result<String> {
    info!("🎯 Routing inference request: {}", request.request_id);

    // Find optimal node based on model, load, and latency requirements
    let optimal_node = state.node_manager
        .find_optimal_node_for_inference(&request).await?;

    // Route request to selected node
    let response = state.node_manager
        .send_inference_request(&optimal_node, request).await?;

    // Update routing metrics
    state.metrics.record_inference_routing(&optimal_node).await;

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_registration() {
        let request = RegisterNodeRequest {
            node_id: "test-node-001".to_string(),
            node_type: "generic".to_string(),
            capabilities: vec!["inference".to_string()],
            hardware_info: HardwareInfo {
                cpu_cores: 4,
                memory_mb: 4096,
                gpu_available: false,
                gpu_memory_mb: None,
                architecture: "x86_64".to_string(),
                power_consumption_watts: Some(25.0),
            },
            location: Some("datacenter-1".to_string()),
        };

        assert!(validate_node_capabilities(&request).is_ok());
    }

    #[test]
    fn test_invalid_node_validation() {
        let request = RegisterNodeRequest {
            node_id: "invalid-node".to_string(),
            node_type: "unknown".to_string(),
            capabilities: vec!["invalid-capability".to_string()],
            hardware_info: HardwareInfo {
                cpu_cores: 0, // Invalid
                memory_mb: 256, // Too low
                gpu_available: false,
                gpu_memory_mb: None,
                architecture: "x86_64".to_string(),
                power_consumption_watts: None,
            },
            location: None,
        };

        assert!(validate_node_capabilities(&request).is_err());
    }
}
