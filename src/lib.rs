//! 🧠 Neural Nexus - Core Library
//! Main library crate for Neural Nexus distributed AI platform

pub mod config;
pub mod orchestrator;
pub mod node;
pub mod inference;
pub mod metrics;

// Re-exports
pub use config::{OrchestratorConfig, NodeConfig};
