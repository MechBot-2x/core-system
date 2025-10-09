"""
🧠 Neural Nexus - Inference Engine
Advanced inference engine with multiple runtime support and optimization features
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# Import inference runtimes
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available")


class InferenceRuntime(Enum):
    """Supported inference runtimes"""
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class DeviceType(Enum):
    """Supported device types"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    NPU = "npu"


@dataclass
class InferenceRequest:
    """Inference request structure"""
    request_id: str
    model_name: str
    input_data: Union[np.ndarray, List[float], Dict[str, np.ndarray]]
    batch_size: Optional[int] = None
    timeout_ms: Optional[int] = None
    priority: int = 5  # 1-10, 10 being highest
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResult:
    """Inference result structure"""
    request_id: str
    outputs: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
    confidence_scores: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    model_version: str = "unknown"
    device_used: str = "unknown"
    batch_size: int = 1
    metadata: Optional[Dict[str, Any]] = None


class ModelLoadError(Exception):
    """Exception raised when model loading fails"""
    pass


class InferenceError(Exception):
    """Exception raised during inference"""
    pass


class BaseInferenceBackend(ABC):
    """Abstract base class for inference backends"""

    def __init__(self, device: DeviceType = DeviceType.CPU):
        self.device = device
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()

    @abstractmethod
    async def load_model(self, model_path: str, model_name: str) -> bool:
        """Load a model from file"""
        pass

    @abstractmethod
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        pass

    @abstractmethod
    async def run_inference(
        self,
        model_name: str,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run inference on the model"""
        pass

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded"""
        with self.lock:
            return model_name in self.models

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models"""
        with self.lock:
            return list(self.models.keys())


class ONNXBackend(BaseInferenceBackend):
    """ONNX Runtime inference backend"""

    def __init__(self, device: DeviceType = DeviceType.CPU):
        super().__init__(device)

        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime is not available")

        # Configure execution providers based on device
        self.providers = self._get_execution_providers()
        logging.info(f"ONNX Runtime providers: {self.providers}")

    def _get_execution_providers(self) -> List[str]:
        """Get available execution providers for ONNX Runtime"""
        providers = []

        if self.device == DeviceType.GPU:
            # Check for CUDA
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            # Check for TensorRT
            if 'TensorrtExecutionProvider' in ort.get_available_providers():
                providers.append('TensorrtExecutionProvider')

        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        return providers

    async def load_model(self, model_path: str, model_name: str) -> bool:
        """Load ONNX model"""
        try:
            logging.info(f"Loading ONNX model: {model_name} from {model_path}")

            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            session = await loop.run_in_executor(
                None,
                lambda: ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=self.providers
                )
            )

            with self.lock:
                self.models[model_name] = session

                # Store metadata
                self.model_metadata[model_name] = {
                    'input_names': [i.name for i in session.get_inputs()],
                    'output_names': [o.name for o in session.get_outputs()],
                    'input_shapes': {i.name: i.shape for i in session.get_inputs()},
                    'output_shapes': {o.name: o.shape for o in session.get_outputs()},
                    'providers': session.get_providers(),
                }

            logging.info(f"✅ ONNX model {model_name} loaded successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to load ONNX model {model_name}: {e}")
            raise ModelLoadError(f"Failed to load ONNX model: {e}")

    async def unload_model(self, model_name: str) -> bool:
        """Unload ONNX model"""
        with self.lock:
            if model_name in self.models:
                del self.models[model_name]
                del self.model_metadata[model_name]
                logging.info(f"ONNX model {model_name} unloaded")
                return True
            return False

    async def run_inference(
        self,
        model_name: str,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run inference using ONNX Runtime"""

        with self.lock:
            if model_name not in self.models:
                raise InferenceError(f"Model {model_name} not loaded")

            session = self.models[model_name]
            metadata = self.model_metadata[model_name]

        try:
            # Prepare inputs
            if isinstance(input_data, dict):
                onnx_inputs = input_data
            else:
                # Single input model
                input_name = metadata['input_names'][0]
                onnx_inputs = {input_name: input_data.astype(np.float32)}

            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: session.run(None, onnx_inputs)
            )

            # Return outputs
            if len(outputs) == 1:
                return outputs[0]
            else:
                return {name: output for name, output in zip(metadata['output_names'], outputs)}

        except Exception as e:
            logging.error(f"ONNX inference failed for {model_name}: {e}")
            raise InferenceError(f"Inference failed: {e}")


class PyTorchBackend(BaseInferenceBackend):
    """PyTorch inference backend"""

    def __init__(self, device: DeviceType = DeviceType.CPU):
        super().__init__(device)

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")

        self.torch_device = self._get_torch_device()
        logging.info(f"PyTorch device: {self.torch_device}")

    def _get_torch_device(self):
        """Get PyTorch device"""
        if self.device == DeviceType.GPU and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    async def load_model(self, model_path: str, model_name: str) -> bool:
        """Load PyTorch model"""
        try:
            logging.info(f"Loading PyTorch model: {model_name} from {model_path}")

            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                lambda: torch.jit.load(model_path, map_location=self.torch_device)
            )

            model.eval()

            with self.lock:
                self.models[model_name] = model
                self.model_metadata[model_name] = {
                    'device': str(self.torch_device),
                    'format': 'torchscript',
                }

            logging.info(f"✅ PyTorch model {model_name} loaded successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to load PyTorch model {model_name}: {e}")
            raise ModelLoadError(f"Failed to load PyTorch model: {e}")

    async def unload_model(self, model_name: str) -> bool:
        """Unload PyTorch model"""
        with self.lock:
            if model_name in self.models:
                del self.models[model_name]
                del self.model_metadata[model_name]
                logging.info(f"PyTorch model {model_name} unloaded")
                return True
            return False

    async def run_inference(
        self,
        model_name: str,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run inference using PyTorch"""

        with self.lock:
            if model_name not in self.models:
                raise InferenceError(f"Model {model_name} not loaded")
            model = self.models[model_name]

        try:
            # Convert to torch tensor
            if isinstance(input_data, dict):
                torch_inputs = {k: torch.from_numpy(v).to(self.torch_device)
                               for k, v in input_data.items()}
            else:
                torch_inputs = torch.from_numpy(input_data).to(self.torch_device)

            # Run inference
            loop = asyncio.get_event_loop()
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None,
                    lambda: model(torch_inputs)
                )

            # Convert back to numpy
            if isinstance(outputs, dict):
                return {k: v.cpu().numpy() for k, v in outputs.items()}
            elif isinstance(outputs, (list, tuple)):
                return [o.cpu().numpy() for o in outputs]
            else:
                return outputs.cpu().numpy()

        except Exception as e:
            logging.error(f"PyTorch inference failed for {model_name}: {e}")
            raise InferenceError(f"Inference failed: {e}")


class InferenceEngine:
    """Main inference engine coordinating multiple backends"""

    def __init__(
        self,
        runtime: InferenceRuntime = InferenceRuntime.ONNX,
        device: DeviceType = DeviceType.CPU,
        max_batch_size: int = 8,
        batch_timeout_ms: int = 100
    ):
        self.runtime = runtime
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        # Initialize backend
        self.backend = self._create_backend(runtime, device)

        # Request queue for batching
        self.request_queue = asyncio.Queue()
        self.batch_processor_task = None

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0

        # Executor for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=4)

        logging.info(f"Inference Engine initialized: {runtime.value} on {device.value}")

    def _create_backend(
        self,
        runtime: InferenceRuntime,
        device: DeviceType
    ) -> BaseInferenceBackend:
        """Create inference backend based on runtime"""

        if runtime == InferenceRuntime.ONNX:
            return ONNXBackend(device)
        elif runtime == InferenceRuntime.PYTORCH:
            return PyTorchBackend(device)
        # Add more backends as needed
        else:
            raise ValueError(f"Unsupported runtime: {runtime}")

    async def load_model(self, model_path: str, model_name: str) -> bool:
        """Load a model"""
        return await self.backend.load_model(model_path, model_name)

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model"""
        return await self.backend.unload_model(model_name)

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return self.backend.is_model_loaded(model_name)

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models"""
        return self.backend.get_loaded_models()

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on a single request"""
        start_time = time.time()

        try:
            self.total_requests += 1

            # Prepare input data
            if isinstance(request.input_data, list):
                input_data = np.array(request.input_data, dtype=np.float32)
            elif isinstance(request.input_data, dict):
                input_data = {k: np.array(v, dtype=np.float32)
                             for k, v in request.input_data.items()}
            else:
                input_data = request.input_data

            # Run inference
            outputs = await self.backend.run_inference(
                request.model_name,
                input_data
            )

            processing_time_ms = (time.time() - start_time) * 1000

            self.successful_requests += 1
            self.total_latency_ms += processing_time_ms

            return InferenceResult(
                request_id=request.request_id,
                outputs=outputs,
                processing_time_ms=processing_time_ms,
                model_version="1.0",
                device_used=self.device.value,
                batch_size=1,
                metadata=request.metadata
            )

        except Exception as e:
            self.failed_requests += 1
            logging.error(f"Inference failed for request {request.request_id}: {e}")
            raise InferenceError(f"Inference failed: {e}")

    async def infer_batch(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Run inference on a batch of requests"""

        if not requests:
            return []

        # Group by model
        model_groups = {}
        for req in requests:
            if req.model_name not in model_groups:
                model_groups[req.model_name] = []
            model_groups[req.model_name].append(req)

        # Process each model group
        results = []
        for model_name, model_requests in model_groups.items():
            # Combine inputs
            if isinstance(model_requests[0].input_data, dict):
                # Multi-input model
                batch_inputs = {}
                for key in model_requests[0].input_data.keys():
                    batch_inputs[key] = np.stack([
                        np.array(req.input_data[key])
                        for req in model_requests
                    ])
            else:
                # Single input model
                batch_inputs = np.stack([
                    np.array(req.input_data) if isinstance(req.input_data, list)
                    else req.input_data
                    for req in model_requests
                ])

            start_time = time.time()

            try:
                # Run batch inference
                batch_outputs = await self.backend.run_inference(
                    model_name,
                    batch_inputs
                )

                processing_time_ms = (time.time() - start_time) * 1000

                # Split outputs back to individual results
                if isinstance(batch_outputs, dict):
                    # Multi-output model
                    for i, req in enumerate(model_requests):
                        outputs = {k: v[i] for k, v in batch_outputs.items()}
                        results.append(InferenceResult(
                            request_id=req.request_id,
                            outputs=outputs,
                            processing_time_ms=processing_time_ms / len(model_requests),
                            model_version="1.0",
                            device_used=self.device.value,
                            batch_size=len(model_requests)
                        ))
                else:
                    # Single output model
                    for i, req in enumerate(model_requests):
                        results.append(InferenceResult(
                            request_id=req.request_id,
                            outputs=batch_outputs[i],
                            processing_time_ms=processing_time_ms / len(model_requests),
                            model_version="1.0",
                            device_used=self.device.value,
                            batch_size=len(model_requests)
                        ))

                self.successful_requests += len(model_requests)

            except Exception as e:
                logging.error(f"Batch inference failed for {model_name}: {e}")
                self.failed_requests += len(model_requests)

                # Return error results
                for req in model_requests:
                    results.append(InferenceResult(
                        request_id=req.request_id,
                        outputs=np.array([]),
                        processing_time_ms=0.0,
                        metadata={'error': str(e)}
                    ))

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        avg_latency = (self.total_latency_ms / self.successful_requests
                      if self.successful_requests > 0 else 0.0)

        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / self.total_requests
                           if self.total_requests > 0 else 0.0),
            'average_latency_ms': avg_latency,
            'loaded_models': self.get_loaded_models(),
            'runtime': self.runtime.value,
            'device': self.device.value,
        }

    async def shutdown(self):
        """Shutdown inference engine"""
        logging.info("Shutting down inference engine...")

        # Unload all models
        for model_name in self.get_loaded_models():
            await self.unload_model(model_name)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logging.info("Inference engine shutdown complete")


# Example usage and testing
async def main():
    """Example usage of the inference engine"""

    logging.basicConfig(level=logging.INFO)

    # Create inference engine
    engine = InferenceEngine(
        runtime=InferenceRuntime.ONNX,
        device=DeviceType.CPU,
        max_batch_size=8
    )

    # Load a model (example)
    model_path = "models/example_model.onnx"
    if Path(model_path).exists():
        await engine.load_model(model_path, "example_model")

        # Create inference request
        request = InferenceRequest(
            request_id="test-001",
            model_name="example_model",
            input_data=np.random.randn(1, 3, 224, 224).astype(np.float32)
        )

        # Run inference
        result = await engine.infer(request)

        print(f"Inference completed in {result.processing_time_ms:.2f}ms")
        print(f"Output shape: {result.outputs.shape}")

        # Print metrics
        metrics = engine.get_metrics()
        print(f"Metrics: {metrics}")

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
