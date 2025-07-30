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
    let status =
