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
pub struct PowerConfig {
    pub management_enabled: bool,
    pub max_power_watts: Option<f32>,
    pub idle_timeout_seconds: u64,
    pub dynamic_scaling: bool,
}

#[derive(Debug, Clone
