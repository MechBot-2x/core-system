// 🧠 Neural Nexus - Edge Node
// Edge computing node for distributed AI inference

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use uuid::Uuid;

// Core modules
mod config;
mod inference_engine;
mod communication;
mod metrics;
mod model_manager;
mod health_monitor;

use config::NodeConfig;
use inference_engine::InferenceEngine;
use communication::{OrchestratorClient, MqttClient};
use metrics::NodeMetrics;
use model_manager::ModelManager;
use health_monitor::HealthMonitor;

/// Edge node state
#[derive(Clone)]
pub struct NodeState {
    pub node_id: String,
    pub config: Arc<NodeConfig>,
    pub inference_engine: Arc<InferenceEngine>,
    pub model_manager: Arc<ModelManager>,
    pub orchestrator_client: Arc<OrchestratorClient>,
    pub mqtt_client: Arc<MqttClient>,
    pub metrics: Arc<NodeMetrics>,
    pub health_monitor: Arc<HealthMonitor>,
    pub status: Arc<RwLock<NodeStatus>>,
}

/// Node status enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeStatus {
    Initializing,
    Registering,
    Healthy,
    Degraded,
    Overloaded,
    Offline,
    Maintenance,
}

/// Inference request structure
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub model_name: String,
    pub input_data: Vec<f32>,
    pub metadata: InferenceMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceMetadata {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source_device: Option<String>,
    pub priority: u8, // 1-10, 10 being highest
    pub max_latency_ms: Option<u64>,
    pub batch_size: Option<usize>,
}

/// Inference response structure
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub results: Vec<f32>,
    pub confidence_scores: Option<Vec<f32>>,
    pub processing_time_ms: u64,
    pub model_version: String,
    pub node_id: String,
}

/// Main edge node execution
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing with structured logging
    tracing_subscriber::fmt()
        .with_env_filter("neural_nexus_node=info,inference=debug")
        .json()
        .init();

    info!("🚀 Starting Neural Nexus Edge Node");

    // Load configuration
    let config = Arc::new(
        NodeConfig::from_env()
            .context("Failed to load node configuration")?
    );

    // Generate or load node ID
    let node_id = config.node.id.clone()
        .unwrap_or_else(|| format!("node-{}", Uuid::new_v4()));

    info!("🏷️ Node ID: {}", node_id);
    info!("🎯 Node Type: {}", config.node.node_type);
    info!("📍 Location: {:?}", config.node.location);

    // Initialize core components
    let inference_engine = Arc::new(
        InferenceEngine::new(&config).await
            .context("Failed to initialize inference engine")?
    );

    let model_manager = Arc::new(
        ModelManager::new(&config, inference_engine.clone()).await
            .context("Failed to initialize model manager")?
    );

    let orchestrator_client = Arc::new(
        OrchestratorClient::new(&config.orchestrator.url).await
            .context("Failed to connect to orchestrator")?
    );

    let mqtt_client = Arc::new(
        MqttClient::new(&config.mqtt).await
            .context("Failed to initialize MQTT client")?
    );

    let metrics = Arc::new(NodeMetrics::new(&node_id));
    
    let health_monitor = Arc::new(
        HealthMonitor::new(&config, metrics.clone())
    );

    info!("🔧 Core components initialized");

    // Create node state
    let state = NodeState {
        node_id: node_id.clone(),
        config: config.clone(),
        inference_engine,
        model_manager,
        orchestrator_client,
        mqtt_client,
        metrics,
        health_monitor,
        status: Arc::new(RwLock::new(NodeStatus::Initializing)),
    };

    // Register with orchestrator
    register_with_orchestrator(&state).await
        .context("Failed to register with orchestrator")?;

    // Start background services
    let heartbeat_task = tokio::spawn(heartbeat_service(state.clone()));
    let metrics_task = tokio::spawn(metrics_collection_service(state.clone()));
    let health_task = tokio::spawn(health_monitoring_service(state.clone()));
    let model_sync_task = tokio::spawn(model_synchronization_service(state.clone()));

    info!("🔄 Background services started");

    // Start main inference server
    let inference_server = tokio::spawn(inference_server_main(state.clone()));

    info!("✅ Neural Nexus Edge Node started successfully!");
    info!("🧠 Inference engine ready on port {}", config.server.port);

    // Wait for shutdown signal or critical error
    tokio::select! {
        result = inference_server => {
            error!("Inference server exited: {:?}", result);
        },
        _ = tokio::signal::ctrl_c() => {
            info!("🛑 Received shutdown signal");
        }
    }

    // Graceful shutdown
    info!("🔄 Initiating graceful shutdown...");
    
    // Update status
    *state.status.write().await = NodeStatus::Offline;
    
    // Notify orchestrator
    if let Err(e) = state.orchestrator_client.unregister_node(&node_id).await {
        warn!("Failed to unregister from orchestrator: {}", e);
    }

    // Cancel background tasks
    heartbeat_task.abort();
    metrics_task.abort();
    health_task.abort();
    model_sync_task.abort();

    // Save final metrics
    if let Err(e) = state.metrics.save_final_snapshot().await {
        warn!("Failed to save final metrics: {}", e);
    }

    info!("✅ Neural Nexus Edge Node shutdown complete");
    Ok(())
}

/// Register node with the orchestrator
async fn register_with_orchestrator(state: &NodeState) -> Result<()> {
    info!("📝 Registering with orchestrator...");

    *state.status.write().await = NodeStatus::Registering;

    let hardware_info = detect_hardware_capabilities().await?;
    let capabilities = determine_node_capabilities(&hardware_info, &state.config).await?;

    let registration_request = crate::RegisterNodeRequest {
        node_id: state.node_id.clone(),
        node_type: state.config.node.node_type.clone(),
        capabilities,
        hardware_info,
        location: state.config.node.location.clone(),
    };

    let registration_token = state.orchestrator_client
        .register_node(registration_request).await
        .context("Orchestrator registration failed")?;

    info!("✅ Successfully registered with orchestrator");
    info!("🔑 Registration token: {}", registration_token);

    *state.status.write().await = NodeStatus::Healthy;
    Ok(())
}

/// Detect hardware capabilities of the current node
async fn detect_hardware_capabilities() -> Result<crate::HardwareInfo> {
    use sysinfo::{System, SystemExt, ProcessorExt};
    
    let mut system = System::new_all();
    system.refresh_all();

    let cpu_cores = system.processors().len() as u32;
    let memory_mb = system.total_memory() / 1024; // Convert to MB
    
    // Try to detect GPU
    let (gpu_available, gpu_memory_mb) = detect_gpu_info().await;
    
    // Detect architecture
    let architecture = std::env::consts::ARCH.to_string();
    
    // Estimate power consumption based on hardware
    let power_consumption_watts = estimate_power_consumption(
        cpu_cores,
        memory_mb,
        gpu_available,
        &architecture
    );

    Ok(crate::HardwareInfo {
        cpu_cores,
        memory_mb,
        gpu_available,
        gpu_memory_mb,
        architecture,
        power_consumption_watts: Some(power_consumption_watts),
    })
}

/// Detect GPU information
async fn detect_gpu_info() -> (bool, Option<u64>) {
    // Try NVIDIA GPU detection
    if let Ok(output) = tokio::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
        .await
    {
        if output.status.success() {
            if let Ok(memory_str) = String::from_utf8(output.stdout) {
                if let Ok(memory_mb) = memory_str.trim().parse::<u64>() {
                    return (true, Some(memory_mb));
                }
            }
        }
    }

    // TODO: Add detection for other GPU types (AMD, Intel, etc.)
    (false, None)
}

/// Estimate power consumption based on hardware
fn estimate_power_consumption(
    cpu_cores: u32,
    memory_mb: u64,
    gpu_available: bool,
    architecture: &str,
) -> f64 {
    let mut base_power = match architecture {
        "aarch64" => 5.0,  // ARM-based (Raspberry Pi, Jetson)
        "x86_64" => 15.0,  // x86 desktop/server
        _ => 10.0,         // Default estimate
    };

    // Add power for additional CPU cores
    base_power += (cpu_cores.saturating_sub(1) as f64) * 2.0;
    
    // Add power for memory (rough estimate)
    base_power += (memory_mb as f64 / 1024.0) * 0.5;
    
    // Add significant power for GPU
    if gpu_available {
        base_power += match architecture {
            "aarch64" => 10.0,  // Jetson integrated GPU
            _ => 75.0,          // Discrete GPU
        };
    }

    base_power
}

/// Determine node capabilities based on hardware and config
async fn determine_node_capabilities(
    hardware: &crate::HardwareInfo,
    config: &NodeConfig,
) -> Result<Vec<String>> {
    let mut capabilities = vec!["inference".to_string()];

    // Add GPU acceleration if available
    if hardware.gpu_available {
        capabilities.push("gpu-acceleration".to_string());
    }

    // Add training capability for powerful nodes
    if hardware.cpu_cores >= 4 && hardware.memory_mb >= 4096 {
        capabilities.push("federated-learning".to_string());
    }

    // Add preprocessing for nodes with sufficient memory
    if hardware.memory_mb >= 2048 {
        capabilities.push("preprocessing".to_string());
    }

    // Always add monitoring
    capabilities.push("monitoring".to_string());

    // Add custom capabilities from config
    if let Some(custom_caps) = &config.node.additional_capabilities {
        capabilities.extend(custom_caps.clone());
    }

    Ok(capabilities)
}

/// Main inference server loop
async fn inference_server_main(state: NodeState) -> Result<()> {
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    let addr = format!("0.0.0.0:{}", state.config.server.port);
    let listener = TcpListener::bind(&addr).await
        .context("Failed to bind inference server")?;

    info!("🧠 Inference server listening on {}", addr);

    while let Ok((stream, addr)) = listener.accept().await {
        let state_clone = state.clone();
        
        tokio::spawn(async move {
            debug!("📡 New connection from: {}", addr);
            
            if let Err(e) = handle_client_connection(state_clone, stream).await {
                warn!("Client connection error: {}", e);
            }
        });
    }

    Ok(())
}

/// Handle individual client connections
async fn handle_client_connection(
    state: NodeState,
    stream: tokio::net::TcpStream,
) -> Result<()> {
    let ws_stream = accept_async(stream).await
        .context("WebSocket handshake failed")?;

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    while let Some(msg) = ws_receiver.next().await {
        match msg? {
            Message::Text(text) => {
                // Parse inference request
                let request: InferenceRequest = serde_json::from_str(&text)
                    .context("Failed to parse inference request")?;

                // Process inference
                let response = process_inference_request(&state, request).await?;

                // Send response
                let response_text = serde_json::to_string(&response)?;
                ws_sender.send(Message::Text(response_text)).await?;
            },
            Message::Binary(data) => {
                // Handle binary inference data (images, audio, etc.)
                let response = process_binary_inference(&state, data).await?;
                let response_bytes = bincode::serialize(&response)?;
                ws_sender.send(Message::Binary(response_bytes)).await?;
            },
            Message::Close(_) => {
                debug!("Client disconnected");
                break;
            },
            _ => {
                // Ignore other message types
            }
        }
    }

    Ok(())
}

/// Process inference request
async fn process_inference_request(
    state: &NodeState,
    request: InferenceRequest,
) -> Result<InferenceResponse> {
    let start_time = std::time::Instant::now();
    
    debug!("🧠 Processing inference request: {}", request.request_id);

    // Update metrics
    state.metrics.increment_inference_requests().await;

    // Load model if not already loaded
    if !state.model_manager.is_model_loaded(&request.model_name).await? {
        state.model_manager.load_model(&request.model_name).await
            .context("Failed to load model")?;
    }

    // Perform inference
    let results = state.inference_engine
        .run_inference(&request.model_name, &request.input_data).await
        .context("Inference execution failed")?;

    let processing_time = start_time.elapsed();
    
    // Update metrics
    state.metrics.record_inference_latency(processing_time).await;
    state.metrics.increment_successful_inferences().await;

    // Create response
    let response = InferenceResponse {
        request_id: request.request_id,
        results: results.outputs,
        confidence_scores: results.confidence_scores,
        processing_time_ms: processing_time.as_millis() as u64,
        model_version: results.model_version,
        node_id: state.node_id.clone(),
    };

    debug!("✅ Inference completed in {}ms", processing_time.as_millis());
    
    Ok(response)
}

/// Process binary inference data
async fn process_binary_inference(
    state: &NodeState,
    data: Vec<u8>,
) -> Result<InferenceResponse> {
    // TODO: Implement binary data processing (images, audio, etc.)
    // This is a placeholder implementation
    
    let request_id = Uuid::new_v4().to_string();
    
    // Convert binary data to float array (placeholder)
    let input_data: Vec<f32> = data.iter()
        .map(|&b| b as f32 / 255.0)
        .collect();

    // Create mock request
    let request = InferenceRequest {
        request_id: request_id.clone(),
        model_name: "default".to_string(),
        input_data,
        metadata: InferenceMetadata {
            timestamp: chrono::Utc::now(),
            source_device: None,
            priority: 5,
            max_latency_ms: None,
            batch_size: None,
        },
    };

    process_inference_request(state, request).await
}

/// Background heartbeat service
async fn heartbeat_service(state: NodeState) {
    let mut interval = tokio::time::interval(
        std::time::Duration::from_secs(state.config.orchestrator.heartbeat_interval)
    );

    loop {
        interval.tick().await;
        
        if let Err(e) = send_heartbeat(&state).await {
            error!("Failed to send heartbeat: {}", e);
            
            // Update status to degraded if heartbeat fails
            *state.status.write().await = NodeStatus::Degraded;
        }
    }
}

/// Send heartbeat to orchestrator
async fn send_heartbeat(state: &NodeState) -> Result<()> {
    let current_metrics = state.metrics.get_current_snapshot().await?;
    let status = state.status.read().await.clone();

    let heartbeat = NodeHeartbeat {
        node_id: state.node_id.clone(),
        timestamp: chrono::Utc::now(),
        status,
        metrics: current_metrics,
        active_models: state.model_manager.get_loaded_models().await?,
    };

    state.orchestrator_client.send_heartbeat(heartbeat).await
        .context("Failed to send heartbeat to orchestrator")?;

    debug!("💓 Heartbeat sent successfully");
    Ok(())
}

/// Node heartbeat structure
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeHeartbeat {
    pub node_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: NodeStatus,
    pub metrics: NodeMetricsSnapshot,
    pub active_models: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeMetricsSnapshot {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: Option<f64>,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub inference_count: u64,
    pub average_latency_ms: f64,
    pub error_count: u64,
    pub uptime_seconds: u64,
}

/// Background metrics collection service
async fn metrics_collection_service(state: NodeState) {
    let mut interval = tokio::time::interval(
        std::time::Duration::from_secs(10)
    );

    loop {
        interval.tick().await;
        
        if let Err(e) = collect_system_metrics(&state).await {
            warn!("Failed to collect system metrics: {}", e);
        }
    }
}

/// Collect comprehensive system metrics
async fn collect_system_metrics(state: &NodeState) -> Result<()> {
    use sysinfo::{System, SystemExt, NetworkExt, ProcessorExt};
    
    let mut system = System::new_all();
    system.refresh_all();

    // CPU metrics
    let cpu_usage = system.global_processor_info().cpu_usage() as f64 / 100.0;
    state.metrics.record_cpu_usage(cpu_usage).await;

    // Memory metrics
    let memory_usage = (system.used_memory() as f64) / (system.total_memory() as f64);
    state.metrics.record_memory_usage(memory_usage).await;

    // Network metrics
    let networks = system.networks();
    let mut total_rx = 0u64;
    let mut total_tx = 0u64;
    
    for (_, network) in networks {
        total_rx += network.received();
        total_tx += network.transmitted();
    }
    
    state.metrics.record_network_io(total_rx, total_tx).await;

    // GPU metrics (if available)
    if let Ok(gpu_usage) = get_gpu_usage().await {
        state.metrics.record_gpu_usage(gpu_usage).await;
    }

    // Temperature metrics (if available)
    if let Ok(temp) = get_cpu_temperature().await {
        state.metrics.record_temperature(temp).await;
    }

    debug!("📊 System metrics collected: CPU={:.1}%, Memory={:.1}%", 
           cpu_usage * 100.0, memory_usage * 100.0);
    
    Ok(())
}

/// Get GPU usage if available
async fn get_gpu_usage() -> Result<f64> {
    let output = tokio::process::Command::new("nvidia-smi")
        .arg("--query-gpu=utilization.gpu")
        .arg("--format=csv,noheader,nounits")
        .output()
        .await?;

    if output.status.success() {
        let usage_str = String::from_utf8(output.stdout)?;
        let usage = usage_str.trim().parse::<f64>()?;
        Ok(usage / 100.0)
    } else {
        anyhow::bail!("Failed to get GPU usage");
    }
}

/// Get CPU temperature if available
async fn get_cpu_temperature() -> Result<f64> {
    // Try to read from thermal zone (Linux)
    if let Ok(temp_str) = tokio::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").await {
        if let Ok(temp_millic) = temp_str.trim().parse::<i32>() {
            return Ok(temp_millic as f64 / 1000.0);
        }
    }

    // Try vcgencmd for Raspberry Pi
    let output = tokio::process::Command::new("vcgencmd")
        .arg("measure_temp")
        .output()
        .await?;

    if output.status.success() {
        let temp_str = String::from_utf8(output.stdout)?;
        if let Some(temp_part) = temp_str.strip_prefix("temp=").and_then(|s| s.strip_suffix("'C\n")) {
            if let Ok(temp) = temp_part.parse::<f64>() {
                return Ok(temp);
            }
        }
    }

    anyhow::bail!("Unable to read CPU temperature");
}

/// Background health monitoring service
async fn health_monitoring_service(state: NodeState) {
    let mut interval = tokio::time::interval(
        std::time::Duration::from_secs(30)
    );

    loop {
        interval.tick().await;
        
        if let Err(e) = perform_health_check(&state).await {
            error!("Health check failed: {}", e);
        }
    }
}

/// Perform comprehensive health check
async fn perform_health_check(state: &NodeState) -> Result<()> {
    let mut current_status = state.status.read().await.clone();
    let metrics = state.metrics.get_current_snapshot().await?;

    // Check CPU usage
    if metrics.cpu_usage > 0.9 {
        warn!("⚠️ High CPU usage detected: {:.1}%", metrics.cpu_usage * 100.0);
        current_status = NodeStatus::Overloaded;
    }

    // Check memory usage
    if metrics.memory_usage > 0.85 {
        warn!("⚠️ High memory usage detected: {:.1}%", metrics.memory_usage * 100.0);
        if current_status == NodeStatus::Healthy {
            current_status = NodeStatus::Degraded;
        }
    }

    // Check error rate
    let error_rate = if metrics.inference_count > 0 {
        metrics.error_count as f64 / metrics.inference_count as f64
    } else {
        0.0
    };

    if error_rate > 0.1 {
        warn!("⚠️ High error rate detected: {:.1}%", error_rate * 100.0);
        current_status = NodeStatus::Degraded;
    }

    // Check orchestrator connectivity
    if let Err(_) = state.orchestrator_client.ping().await {
        warn!("⚠️ Lost connection to orchestrator");
        current_status = NodeStatus::Degraded;
    }

    // Check disk space
    if let Ok(disk_usage) = get_disk_usage().await {
        if disk_usage > 0.9 {
            warn!("⚠️ Low disk space: {:.1}% used", disk_usage * 100.0);
            current_status = NodeStatus::Degraded;
        }
    }

    // Update status if changed
    let mut status_guard = state.status.write().await;
    if *status_guard != current_status {
        info!("🔄 Node status changed: {:?} -> {:?}", *status_guard, current_status);
        *status_guard = current_status;
    }

    Ok(())
}

/// Get disk usage percentage
async fn get_disk_usage() -> Result<f64> {
    use std::path::Path;
    
    let statvfs = nix::sys::statvfs::statvfs(Path::new("/"))?;
    let total_space = statvfs.blocks() * statvfs.fragment_size();
    let available_space = statvfs.blocks_available() * statvfs.fragment_size();
    let used_space = total_space - available_space;
    
    Ok(used_space as f64 / total_space as f64)
}

/// Background model synchronization service
async fn model_synchronization_service(state: NodeState) {
    let mut interval = tokio::time::interval(
        std::time::Duration::from_secs(300) // Check every 5 minutes
    );

    loop {
        interval.tick().await;
        
        if let Err(e) = synchronize_models(&state).await {
            warn!("Model synchronization failed: {}", e);
        }
    }
}

/// Synchronize models with orchestrator
async fn synchronize_models(state: &NodeState) -> Result<()> {
    debug!("🔄 Synchronizing models with orchestrator");

    // Get available models from orchestrator
    let available_models = state.orchestrator_client
        .get_available_models().await?;

    // Get currently loaded models
    let loaded_models = state.model_manager.get_loaded_models().await?;

    // Check for new models to download
    for model_info in &available_models {
        if !loaded_models.contains(&model_info.name) {
            info!("📥 Downloading new model: {}", model_info.name);
            
            if let Err(e) = state.model_manager
                .download_model(&model_info.name, &model_info.url).await {
                error!("Failed to download model {}: {}", model_info.name, e);
            }
        }
    }

    // Check for outdated models to update
    for model_name in &loaded_models {
        if let Some(model_info) = available_models.iter()
            .find(|m| m.name == *model_name) {
            
            let local_version = state.model_manager
                .get_model_version(model_name).await?;
            
            if local_version != model_info.version {
                info!("🔄 Updating model: {} {} -> {}", 
                      model_name, local_version, model_info.version);
                      
                if let Err(e) = state.model_manager
                    .update_model(model_name, &model_info.url).await {
                    error!("Failed to update model {}: {}", model_name, e);
                }
            }
        }
    }

    debug!("✅ Model synchronization completed");
    Ok(())
}

/// Graceful shutdown handler
pub async fn shutdown_node(state: &NodeState) -> Result<()> {
    info!("🔄 Starting graceful shutdown...");

    // Stop accepting new requests
    *state.status.write().await = NodeStatus::Maintenance;

    // Wait for current inferences to complete (with timeout)
    let shutdown_timeout = tokio::time::Duration::from_secs(30);
    let shutdown_deadline = tokio::time::Instant::now() + shutdown_timeout;

    while state.metrics.get_active_inference_count().await > 0 {
        if tokio::time::Instant::now() > shutdown_deadline {
            warn!("⚠️ Shutdown timeout reached, forcing shutdown");
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Unload all models
    if let Err(e) = state.model_manager.unload_all_models().await {
        warn!("Failed to unload models: {}", e);
    }

    // Save final metrics
    if let Err(e) = state.metrics.save_shutdown_metrics().await {
        warn!("Failed to save shutdown metrics: {}", e);
    }

    // Update final status
    *state.status.write().await = NodeStatus::Offline;

    info!("✅ Node shutdown completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hardware_detection() {
        let hardware = detect_hardware_capabilities().await.unwrap();
        
        assert!(hardware.cpu_cores > 0);
        assert!(hardware.memory_mb > 0);
        assert!(!hardware.architecture.is_empty());
    }

    #[tokio::test]
    async fn test_node_capabilities() {
        let hardware = crate::HardwareInfo {
            cpu_cores: 4,
            memory_mb: 4096,
            gpu_available: true,
            gpu_memory_mb: Some(8192),
            architecture: "x86_64".to_string(),
            power_consumption_watts: Some(50.0),
        };

        let config = NodeConfig::default();
        let capabilities = determine_node_capabilities(&hardware, &config).await.unwrap();

        assert!(capabilities.contains(&"inference".to_string()));
        assert!(capabilities.contains(&"gpu-acceleration".to_string()));
        assert!(capabilities.contains(&"federated-learning".to_string()));
    }

    #[test]
    fn test_power_estimation() {
        let power = estimate_power_consumption(4, 8192, true, "x86_64");
        assert!(power > 20.0);
        assert!(power < 200.0);

        let arm_power = estimate_power_consumption(4, 4096, false, "aarch64");
        assert!(arm_power < power); // ARM should use less power
    }
}
