// 🧠 Neural Nexus - Configuration Management
// Centralized configuration system for orchestrator and nodes

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Main orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub orchestrator: OrchestratorSettings,
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub grpc: GrpcConfig,
    pub http: HttpConfig,
    pub mqtt: MqttConfig,
    pub monitoring: MonitoringConfig,
    pub security: SecurityConfig,
    pub logging: LoggingConfig,
}

/// Core orchestrator settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorSettings {
    pub bind_address: String,
    pub node_timeout_seconds: u64,
    pub max_nodes: usize,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub auto_scaling: bool,
    pub backup_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    ResourceBased,
    LatencyBased,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub connection_timeout_seconds: u64,
    pub migration_enabled: bool,
    pub backup_schedule: Option<String>, // Cron expression
}

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub max_connections: u32,
    pub connection_timeout_seconds: u64,
    pub key_prefix: String,
    pub ttl_seconds: u64,
}

/// gRPC server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    pub bind_address: String,
    pub max_message_size: usize,
    pub max_concurrent_streams: u32,
    pub keepalive_interval_seconds: u64,
    pub tls_enabled: bool,
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
}

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    pub bind_address: String,
    pub request_timeout_seconds: u64,
    pub max_payload_size: usize,
    pub cors_enabled: bool,
    pub cors_origins: Vec<String>,
    pub rate_limiting: RateLimitConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_minute: u32,
    pub burst_size: u32,
}

/// MQTT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttConfig {
    pub broker_url: String,
    pub client_id: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub keep_alive_seconds: u16,
    pub clean_session: bool,
    pub qos: u8,
    pub topics: MqttTopics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttTopics {
    pub heartbeat: String,
    pub inference_requests: String,
    pub inference_responses: String,
    pub model_updates: String,
    pub system_alerts: String,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub prometheus_enabled: bool,
    pub prometheus_bind_address: String,
    pub metrics_interval_seconds: u64,
    pub health_check_enabled: bool,
    pub health_check_interval_seconds: u64,
    pub alerting: AlertingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub enabled: bool,
    pub webhook_url: Option<String>,
    pub email_notifications: Option<EmailConfig>,
    pub slack_webhook: Option<String>,
    pub thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    pub smtp_server: String,
    pub smtp_port: u16,
    pub username: String,
    pub password: String,
    pub from_address: String,
    pub to_addresses: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub error_rate_percent: f64,
    pub response_time_ms: u64,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub tls_enabled: bool,
    pub mtls_enabled: bool,
    pub ca_cert_path: Option<PathBuf>,
    pub server_cert_path: Option<PathBuf>,
    pub server_key_path: Option<PathBuf>,
    pub jwt_secret: String,
    pub jwt_expiry_hours: u64,
    pub api_key_required: bool,
    pub allowed_origins: Vec<String>,
    pub encryption: EncryptionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub data_at_rest: bool,
    pub data_in_transit: bool,
    pub differential_privacy: bool,
    pub homomorphic_encryption: bool,
    pub key_rotation_days: u32,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: LogFormat,
    pub output: LogOutput,
    pub rotation: LogRotation,
    pub structured_logging: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Text,
    Json,
    Compact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    Stdout,
    File(PathBuf),
    Syslog,
    Combined(Vec<LogOutput>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotation {
    pub enabled: bool,
    pub max_size_mb: u64,
    pub max_files: u32,
    pub compress: bool,
}

/// Edge node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub node: NodeSettings,
    pub orchestrator: OrchestratorClientConfig,
    pub inference: InferenceConfig,
    pub models: ModelConfig,
    pub hardware: HardwareConfig,
    pub networking: NetworkingConfig,
    pub monitoring: NodeMonitoringConfig,
    pub mqtt: MqttConfig,
    pub server: ServerConfig,
    pub security: NodeSecurityConfig,
    pub logging: LoggingConfig,
}

/// Node-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSettings {
    pub id: Option<String>,
    pub node_type: String, // raspberry-pi, jetson, generic, server
    pub location: Option<String>,
    pub description: Option<String>,
    pub tags: HashMap<String, String>,
    pub additional_capabilities: Option<Vec<String>>,
    pub auto_register: bool,
    pub auto_update: bool,
}

/// Orchestrator client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorClientConfig {
    pub url: String,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub retry_delay_seconds: u64,
    pub heartbeat_interval: u64,
    pub registration_timeout_seconds: u64,
}

/// Inference engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub runtime: InferenceRuntime,
    pub device: InferenceDevice,
    pub optimization: OptimizationConfig,
    pub batching: BatchingConfig,
    pub caching: CachingConfig,
    pub preprocessing: PreprocessingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceRuntime {
    OnnxRuntime,
    TensorRT,
    TensorFlowLite,
    PyTorch,
    OpenVINO,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceDevice {
    CPU,
    GPU,
    TPU,
    NPU, // Neural Processing Unit
    Auto, // Auto-detect best device
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub quantization: QuantizationConfig,
    pub pruning: PruningConfig,
    pub graph_optimization: bool,
    pub memory_optimization: bool,
    pub parallelization: ParallelizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub precision: QuantizationPrecision,
    pub calibration_dataset_size: Option<usize>,
    pub dynamic_quantization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    INT8,
    INT16,
    FP16,
    FP32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub enabled: bool,
    pub sparsity_ratio: f32,
    pub structured_pruning: bool,
    pub gradual_pruning: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationConfig {
    pub enabled: bool,
    pub thread_count: Option<usize>,
    pub inter_op_threads: Option<usize>,
    pub intra_op_threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    pub enabled: bool,
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
    pub dynamic_batching: bool,
    pub padding_strategy: PaddingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingStrategy {
    None,
    ZeroPadding,
    ReplicationPadding,
    ReflectionPadding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    pub enabled: bool,
    pub max_cache_size_mb: usize,
    pub cache_ttl_seconds: u64,
    pub cache_strategy: CacheStrategy,
    pub persistent_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStrategy {
    LRU, // Least Recently Used
    LFU, // Least Frequently Used
    FIFO, // First In, First Out
    TTL, // Time To Live
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub enabled: bool,
    pub normalization: NormalizationConfig,
    pub resizing: ResizingConfig,
    pub augmentation: AugmentationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub enabled: bool,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizingConfig {
    pub enabled: bool,
    pub width: u32,
    pub height: u32,
    pub interpolation: InterpolationMethod,
    pub maintain_aspect_ratio: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    Nearest,
    Linear,
    Cubic,
    Lanczos,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    pub enabled: bool,
    pub random_rotation: Option<f32>,
    pub random_flip: bool,
    pub random_brightness: Option<f32>,
    pub random_contrast: Option<f32>,
}

/// Model management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub storage_path: PathBuf,
    pub auto_download: bool,
    pub auto_update: bool,
    pub checksum_validation: bool,
    pub compression: CompressionConfig,
    pub versioning: VersioningConfig,
    pub registry: ModelRegistryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub algorithm: CompressionAlgorithm,
    pub compression_level: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Lz4,
    Zstd,
    Brotli,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningConfig {
    pub enabled: bool,
    pub max_versions: u32,
    pub auto_cleanup: bool,
    pub rollback_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryConfig {
    pub url: String,
    pub authentication: RegistryAuthentication,
    pub sync_interval_seconds: u64,
    pub cache_metadata: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryAuthentication {
    pub method: AuthMethod,
    pub credentials: AuthCredentials,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    None,
    BasicAuth,
    BearerToken,
    ApiKey,
    OAuth2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCredentials {
    pub username: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
    pub api_key: Option<String>,
}

/// Hardware-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub cpu: CpuConfig,
    pub memory: MemoryConfig,
    pub gpu: Option<GpuConfig>,
    pub storage: StorageConfig,
    pub power: PowerConfig,
    pub thermal: ThermalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    pub affinity: Option<Vec<usize>>, // CPU cores to use
    pub governor: Option<String>, // CPU frequency governor
    pub max_frequency_mhz: Option<u32>,
    pub performance_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_usage_mb: Option<usize>,
    pub swap_enabled: bool,
    pub memory_mapping: bool,
    pub huge_pages: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub device_id: Option<u32>,
    pub memory_limit_mb: Option<usize>,
    pub compute_mode: ComputeMode,
    pub power_limit_watts: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeMode {
    Default,
    Exclusive,
    Prohibited,
    ExclusiveProcess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_path: PathBuf,
    pub temp_path: PathBuf,
    pub max_disk_usage_percent: f32,
    pub cleanup_policy: CleanupPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupPolicy {
    pub enabled: bool,
    pub max_age_days: u32,
    pub max_size_gb: f32,
    pub cleanup_interval_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    pub monitoring_enabled: bool,
    pub max_temperature_celsius: f32,
    pub throttling_enabled: bool,
    pub fan_control: FanControlConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanControlConfig {
    pub enabled: bool,
    pub auto_control: bool,
    pub min_speed_percent: u8,
    pub max_speed_percent: u8,
    pub temperature_curve: Vec<(f32, u8)>, // (temp, speed) pairs
}

/// Networking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkingConfig {
    pub interface: Option<String>,
    pub bandwidth_limit_mbps: Option<u32>,
    pub quality_of_service: QosConfig,
    pub compression: NetworkCompressionConfig,
    pub retry_policy: RetryPolicyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosConfig {
    pub enabled: bool,
    pub priority: NetworkPriority,
    pub buffer_size_kb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCompressionConfig {
    pub enabled: bool,
    pub algorithm: NetworkCompressionAlgorithm,
    pub compression_level: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkCompressionAlgorithm {
    Gzip,
    Deflate,
    Brotli,
    Lz4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicyConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f32,
    pub jitter: bool,
}

/// Node monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMonitoringConfig {
    pub enabled: bool,
    pub metrics_interval_seconds: u64,
    pub health_check_interval_seconds: u64,
    pub system_metrics: SystemMetricsConfig,
    pub performance_metrics: PerformanceMetricsConfig,
    pub custom_metrics: HashMap<String, MetricConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsConfig {
    pub cpu_usage: bool,
    pub memory_usage: bool,
    pub disk_usage: bool,
    pub network_io: bool,
    pub gpu_usage: bool,
    pub temperature: bool,
    pub power_consumption: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsConfig {
    pub inference_latency: bool,
    pub throughput: bool,
    pub error_rate: bool,
    pub queue_depth: bool,
    pub batch_efficiency: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricConfig {
    pub enabled: bool,
    pub collection_interval_seconds: u64,
    pub aggregation: MetricAggregation,
    pub retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricAggregation {
    Average,
    Sum,
    Min,
    Max,
    Count,
    Percentile(f32),
}

/// Server configuration for edge nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub port: u16,
    pub bind_address: String,
    pub max_connections: u32,
    pub connection_timeout_seconds: u64,
    pub request_timeout_seconds: u64,
    pub websocket_enabled: bool,
    pub http_enabled: bool,
    pub grpc_enabled: bool,
}

/// Node security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSecurityConfig {
    pub certificate_path: Option<PathBuf>,
    pub private_key_path: Option<PathBuf>,
    pub ca_certificate_path: Option<PathBuf>,
    pub mutual_tls: bool,
    pub api_key: Option<String>,
    pub secure_boot: bool,
    pub attestation: AttestationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationConfig {
    pub enabled: bool,
    pub tpm_enabled: bool,
    pub secure_enclave: bool,
    pub measurement_validation: bool,
}

// Implementation methods for configuration loading
impl OrchestratorConfig {
    /// Load orchestrator configuration from environment variables and files
    pub fn from_env() -> Result<Self> {
        let config_path = std::env::var("NEURAL_NEXUS_CONFIG")
            .unwrap_or_else(|_| "config/orchestrator.toml".to_string());

        Self::from_file(&config_path)
    }

    /// Load configuration from TOML file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;

        let mut config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path))?;

        // Override with environment variables
        config.override_from_env()?;

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Override configuration values with environment variables
    fn override_from_env(&mut self) -> Result<()> {
        // Database URL
        if let Ok(db_url) = std::env::var("DATABASE_URL") {
            self.database.url = db_url;
        }

        // Redis URL
        if let Ok(redis_url) = std::env::var("REDIS_URL") {
            self.redis.url = redis_url;
        }

        // Bind addresses
        if let Ok(bind_addr) = std::env::var("ORCHESTRATOR_BIND_ADDRESS") {
            self.orchestrator.bind_address = bind_addr;
        }

        if let Ok(grpc_addr) = std::env::var("GRPC_BIND_ADDRESS") {
            self.grpc.bind_address = grpc_addr;
        }

        if let Ok(http_addr) = std::env::var("HTTP_BIND_ADDRESS") {
            self.http.bind_address = http_addr;
        }

        // Security settings
        if let Ok(jwt_secret) = std::env::var("JWT_SECRET") {
            self.security.jwt_secret = jwt_secret;
        }

        // MQTT settings
        if let Ok(mqtt_broker) = std::env::var("MQTT_BROKER_URL") {
            self.mqtt.broker_url = mqtt_broker;
        }

        if let Ok(mqtt_user) = std::env::var("MQTT_USERNAME") {
            self.mqtt.username = Some(mqtt_user);
        }

        if let Ok(mqtt_pass) = std::env::var("MQTT_PASSWORD") {
            self.mqtt.password = Some(mqtt_pass);
        }

        Ok(())
    }

    /// Validate configuration values
    fn validate(&self) -> Result<()> {
        // Validate bind addresses
        if self.orchestrator.bind_address.is_empty() {
            anyhow::bail!("Orchestrator bind address cannot be empty");
        }

        if self.grpc.bind_address.is_empty() {
            anyhow::bail!("gRPC bind address cannot be empty");
        }

        if self.http.bind_address.is_empty() {
            anyhow::bail!("HTTP bind address cannot be empty");
        }

        // Validate database URL
        if self.database.url.is_empty() {
            anyhow::bail!("Database URL cannot be empty");
        }

        // Validate security settings
        if self.security.jwt_secret.len() < 32 {
            anyhow::bail!("JWT secret must be at least 32 characters long");
        }

        // Validate resource limits
        if self.orchestrator.max_nodes == 0 {
            anyhow::bail!("Max nodes must be greater than 0");
        }

        Ok(())
    }

    /// Get default configuration for development
    pub fn default_dev() -> Self {
        Self {
            orchestrator: OrchestratorSettings {
                bind_address: "0.0.0.0:8080".to_string(),
                node_timeout_seconds: 300,
                max_nodes: 1000,
                load_balancing_strategy: LoadBalancingStrategy::ResourceBased,
                auto_scaling: true,
                backup_enabled: false,
            },
            database: DatabaseConfig {
                url: "postgresql://postgres:password@localhost:5432/neural_nexus".to_string(),
                max_connections: 10,
                connection_timeout_seconds: 30,
                migration_enabled: true,
                backup_schedule: None,
            },
            redis: RedisConfig {
                url: "redis://localhost:6379".to_string(),
                max_connections: 10,
                connection_timeout_seconds: 5,
                key_prefix: "nn:".to_string(),
                ttl_seconds: 3600,
            },
            grpc: GrpcConfig {
                bind_address: "0.0.0.0:50051".to_string(),
                max_message_size: 4 * 1024 * 1024, // 4MB
                max_concurrent_streams: 100,
                keepalive_interval_seconds: 30,
                tls_enabled: false,
                cert_path: None,
                key_path: None,
            },
            http: HttpConfig {
                bind_address: "0.0.0.0:8080".to_string(),
                request_timeout_seconds: 30,
                max_payload_size: 10 * 1024 * 1024, // 10MB
                cors_enabled: true,
                cors_origins: vec!["*".to_string()],
                rate_limiting: RateLimitConfig {
                    enabled: false,
                    requests_per_minute: 100,
                    burst_size: 10,
                },
            },
            mqtt: MqttConfig {
                broker_url: "mqtt://localhost:1883".to_string(),
                client_id: "neural-nexus-orchestrator".to_string(),
                username: None,
                password: None,
                keep_alive_seconds: 60,
                clean_session: true,
                qos: 1,
                topics: MqttTopics {
                    heartbeat: "neural-nexus/heartbeat".to_string(),
                    inference_requests: "neural-nexus/inference/requests".to_string(),
                    inference_responses: "neural-nexus/inference/responses".to_string(),
                    model_updates: "neural-nexus/models/updates".to_string(),
                    system_alerts: "neural-nexus/alerts".to_string(),
                },
            },
            monitoring: MonitoringConfig {
                prometheus_enabled: true,
                prometheus_bind_address: "0.0.0.0:9090".to_string(),
                metrics_interval_seconds: 10,
                health_check_enabled: true,
                health_check_interval_seconds: 30,
                alerting: AlertingConfig {
                    enabled: false,
                    webhook_url: None,
                    email_notifications: None,
                    slack_webhook: None,
                    thresholds: AlertThresholds {
                        cpu_usage_percent: 80.0,
                        memory_usage_percent: 85.0,
                        error_rate_percent: 5.0,
                        response_time_ms: 1000,
                    },
                },
            },
            security: SecurityConfig {
                tls_enabled: false,
                mtls_enabled: false,
                ca_cert_path: None,
                server_cert_path: None,
                server_key_path: None,
                jwt_secret: "your-256-bit-secret-key-here-change-in-production".to_string(),
                jwt_expiry_hours: 24,
                api_key_required: false,
                allowed_origins: vec!["*".to_string()],
                encryption: EncryptionConfig {
                    data_at_rest: false,
                    data_in_transit: false,
                    differential_privacy: false,
                    homomorphic_encryption: false,
                    key_rotation_days: 30,
                },
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: LogFormat::Json,
                output: LogOutput::Stdout,
                rotation: LogRotation {
                    enabled: false,
                    max_size_mb: 100,
                    max_files: 5,
                    compress: true,
                },
                structured_logging: true,
            },
        }
    }
}

impl NodeConfig {
    /// Load node configuration from environment variables and files
    pub fn from_env() -> Result<Self> {
        let config_path = std::env::var("NEURAL_NEXUS_NODE_CONFIG")
            .unwrap_or_else(|_| "config/node.toml".to_string());

        Self::from_file(&config_path)
    }

    /// Load configuration from TOML file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read node config file: {}", path))?;

        let mut config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse node config file: {}", path))?;

        // Override with environment variables
        config.override_from_env()?;

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Override configuration values with environment variables
    fn override_from_env(&mut self) -> Result<()> {
        // Node settings
        if let Ok(node_id) = std::env::var("NODE_ID") {
            self.node.id = Some(node_id);
        }

        if let Ok(node_type) = std::env::var("NODE_TYPE") {
            self.node.node_type = node_type;
        }

        if let Ok(location) = std::env::var("NODE_LOCATION") {
            self.node.location = Some(location);
        }

        // Orchestrator settings
        if let Ok(orchestrator_url) = std::env::var("ORCHESTRATOR_URL") {
            self.orchestrator.url = orchestrator_url;
        }

        // Server settings
        if let Ok(port) = std::env::var("NODE_PORT") {
            self.server.port = port.parse()
                .context("Invalid NODE_PORT value")?;
        }

        // MQTT settings
        if let Ok(mqtt_broker) = std::env::var("MQTT_BROKER_URL") {
            self.mqtt.broker_url = mqtt_broker;
        }

        // Model storage path
        if let Ok(model_path) = std::env::var("MODEL_STORAGE_PATH") {
            self.models.storage_path = PathBuf::from(model_path);
        }

        Ok(())
    }

    /// Validate node configuration
    fn validate(&self) -> Result<()> {
        // Validate node type
        match self.node.node_type.as_str() {
            "raspberry-pi" | "jetson" | "generic" | "server" => {},
            _ => anyhow::bail!("Invalid node type: {}", self.node.node_type),
        }

        // Validate orchestrator URL
        if self.orchestrator.url.is_empty() {
            anyhow::bail!("Orchestrator URL cannot be empty");
        }

        // Validate server port
        if self.server.port == 0 {
            anyhow::bail!("Server port must be greater than 0");
        }

        // Validate model storage path
        if !self.models.storage_path.exists() {
            std::fs::create_dir_all(&self.models.storage_path)
                .with_context(|| {
                    format!("Failed to create model storage directory: {:?}", 
                           self.models.storage_path)
                })?;
        }

        Ok(())
    }

    /// Get default configuration for development
    pub fn default() -> Self {
        Self {
            node: NodeSettings {
                id: None, // Will be auto-generated
                node_type: "generic".to_string(),
                location: None,
                description: Some("Neural Nexus Edge Node".to_string()),
                tags: HashMap::new(),
                additional_capabilities: None,
                auto_register: true,
                auto_update: true,
            },
            orchestrator: OrchestratorClientConfig {
                url: "http://localhost:8080".to_string(),
                timeout_seconds: 30,
                retry_attempts: 3,
                retry_delay_seconds: 5,
                heartbeat_interval: 30,
                registration_timeout_seconds: 60,
            },
            inference: InferenceConfig {
                runtime: InferenceRuntime::OnnxRuntime,
                device: InferenceDevice::Auto,
                optimization: OptimizationConfig {
                    quantization: QuantizationConfig {
                        enabled: false,
                        precision: QuantizationPrecision::FP32,
                        calibration_dataset_size: None,
                        dynamic_quantization: false,
                    },
                    pruning: PruningConfig {
                        enabled: false,
                        sparsity_ratio: 0.0,
                        structured_pruning: false,
                        gradual_pruning: false,
                    },
                    graph_optimization: true,
                    memory_optimization: true,
                    parallelization: ParallelizationConfig {
                        enabled: true,
                        thread_count: None, // Auto-detect
                        inter_op_threads: None,
                        intra_op_threads: None,
                    },
                },
                batching: BatchingConfig {
                    enabled: true,
                    max_batch_size: 8,
                    batch_timeout_ms: 100,
                    dynamic_batching: true,
                    padding_strategy: PaddingStrategy::ZeroPadding,
                },
                caching: CachingConfig {
                    enabled: true,
                    max_cache_size_mb: 1024,
                    cache_ttl_seconds: 3600,
                    cache_strategy: CacheStrategy::LRU,
                    persistent_cache: false,
                },
                preprocessing: PreprocessingConfig {
                    enabled: true,
                    normalization: NormalizationConfig {
                        enabled: false,
                        mean: vec![0.485, 0.456, 0.406],
                        std: vec![0.229, 0.224, 0.225],
                        scale: 1.0,
                    },
                    resizing: ResizingConfig {
                        enabled: false,
                        width: 224,
                        height: 224,
                        interpolation: InterpolationMethod::Linear,
                        maintain_aspect_ratio: false,
                    },
                    augmentation: AugmentationConfig {
                        enabled: false,
                        random_rotation: None,
                        random_flip: false,
                        random_brightness: None,
                        random_contrast: None,
                    },
                },
            },
            models: ModelConfig {
                storage_path: PathBuf::from("./models"),
                auto_download: true,
                auto_update: false,
                checksum_validation: true,
                compression: CompressionConfig {
                    enabled: true,
                    algorithm: CompressionAlgorithm::Gzip,
                    compression_level: 6,
                },
                versioning: VersioningConfig {
                    enabled: true,
                    max_versions: 3,
                    auto_cleanup: true,
                    rollback_enabled: true,
                },
                registry: ModelRegistryConfig {
                    url: "http://localhost:8080/models".to_string(),
                    authentication: RegistryAuthentication {
                        method: AuthMethod::None,
                        credentials: AuthCredentials {
                            username: None,
                            password: None,
                            token: None,
                            api_key: None,
                        },
                    },
                    sync_interval_seconds: 300,
                    cache_metadata: true,
                },
            },
            hardware: HardwareConfig {
                cpu: CpuConfig {
                    affinity: None,
                    governor: None,
                    max_frequency_mhz: None,
                    performance_mode: false,
                },
                memory: MemoryConfig {
                    max_usage_mb: None,
                    swap_enabled: true,
                    memory_mapping: true,
                    huge_pages: false,
                },
                gpu: None, // Will be auto-detected
                storage: StorageConfig {
                    data_path: PathBuf::from("./data"),
                    temp_path: PathBuf::from("/tmp/neural-nexus"),
                    max_disk_usage_percent: 80.0,
                    cleanup_policy: CleanupPolicy {
                        enabled: true,
                        max_age_days: 7,
                        max_size_gb: 10.0,
                        cleanup_interval_hours: 24,
                    },
                },
                power: PowerConfig {
                    management_enabled: false,
                    max_power_watts: None,
                    idle_timeout_seconds: 300,
                    dynamic_scaling: false,
                },
                thermal: ThermalConfig {
                    monitoring_enabled: true,
                    max_temperature_celsius: 85.0,
                    throttling_enabled: true,
                    fan_control: FanControlConfig {
                        enabled: false,
                        auto_control: true,
                        min_speed_percent: 20,
                        max_speed_percent: 100,
                        temperature_curve: vec![
                            (40.0, 20),
                            (60.0, 50),
                            (80.0, 100),
                        ],
                    },
                },
            },
            networking: NetworkingConfig {
                interface: None, // Auto-detect
                bandwidth_limit_mbps: None,
                quality_of_service: QosConfig {
                    enabled: false,
                    priority: NetworkPriority::Normal,
                    buffer_size_kb: 64,
                },
                compression: NetworkCompressionConfig {
                    enabled: true,
                    algorithm: NetworkCompressionAlgorithm::Gzip,
                    compression_level: 6,
                },
                retry_policy: RetryPolicyConfig {
                    max_retries: 3,
                    initial_delay_ms: 1000,
                    max_delay_ms: 30000,
                    backoff_multiplier: 2.0,
                    jitter: true,
                },
            },
            monitoring: NodeMonitoringConfig {
                enabled: true,
                metrics_interval_seconds: 10,
                health_check_interval_seconds: 30,
                system_metrics: SystemMetricsConfig {
                    cpu_usage: true,
                    memory_usage: true,
                    disk_usage: true,
                    network_io: true,
                    gpu_usage: true,
                    temperature: true,
                    power_consumption: false,
                },
                performance_metrics: PerformanceMetricsConfig {
                    inference_latency: true,
                    throughput: true,
                    error_rate: true,
                    queue_depth: true,
                    batch_efficiency: true,
                },
                custom_metrics: HashMap::new(),
            },
            mqtt: MqttConfig {
                broker_url: "mqtt://localhost:1883".to_string(),
                client_id: "neural-nexus-node".to_string(),
                username: None,
                password: None,
                keep_alive_seconds: 60,
                clean_session: true,
                qos: 1,
                topics: MqttTopics {
                    heartbeat: "neural-nexus/heartbeat".to_string(),
                    inference_requests: "neural-nexus/inference/requests".to_string(),
                    inference_responses: "neural-nexus/inference/responses".to_string(),
                    model_updates: "neural-nexus/models/updates".to_string(),
                    system_alerts: "neural-nexus/alerts".to_string(),
                },
            },
            server: ServerConfig {
                port: 8081,
                bind_address: "0.0.0.0".to_string(),
                max_connections: 100,
                connection_timeout_seconds: 30,
                request_timeout_seconds: 30,
                websocket_enabled: true,
                http_enabled: true,
                grpc_enabled: false,
            },
            security: NodeSecurityConfig {
                certificate_path: None,
                private_key_path: None,
                ca_certificate_path: None,
                mutual_tls: false,
                api_key: None,
                secure_boot: false,
                attestation: AttestationConfig {
                    enabled: false,
                    tmp_enabled: false,
                    secure_enclave: false,
                    measurement_validation: false,
                },
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: LogFormat::Json,
                output: LogOutput::Stdout,
                rotation: LogRotation {
                    enabled: false,
                    max_size_mb: 100,
                    max_files: 5,
                    compress: true,
                },
                structured_logging: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_config_validation() {
        let mut config = OrchestratorConfig::default_dev();
        assert!(config.validate().is_ok());

        // Test invalid configuration
        config.orchestrator.bind_address = "".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_node_config_validation() {
        let mut config = NodeConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid node type
        config.node.node_type = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = OrchestratorConfig::default_dev();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: OrchestratorConfig = toml::from_str(&serialized).unwrap();
        
        assert_eq!(config.orchestrator.bind_address, deserialized.orchestrator.bind_address);
    }
}
