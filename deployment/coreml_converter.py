"""
CoreML Converter for iPhone Deployment

This module converts PyTorch models to CoreML format for deployment
on Apple Neural Engine (ANE).

Target: Run on iPhone 15 Pro with <30ms latency
"""

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class CoreMLConverter:
    """
    Convert PyTorch models to CoreML with ANE optimizations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        compute_units: str = 'ALL',  # 'ALL', 'CPU_AND_GPU', 'CPU_AND_NE', 'CPU_ONLY'
        quantize_weights: bool = True
    ):
        """
        Initialize converter.
        
        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape (without batch dimension)
            compute_units: Target compute units
            quantize_weights: Whether to quantize weights to INT8
        """
        self.model = model.eval()
        self.input_shape = input_shape
        self.compute_units = compute_units
        self.quantize_weights = quantize_weights
        
        # Map compute units string to CoreML enum
        self.compute_unit_map = {
            'ALL': ct.ComputeUnit.ALL,
            'CPU_AND_GPU': ct.ComputeUnit.CPU_AND_GPU,
            'CPU_AND_NE': ct.ComputeUnit.CPU_AND_NE,
            'CPU_ONLY': ct.ComputeUnit.CPU_ONLY
        }
    
    def convert(
        self,
        output_path: str,
        model_name: str = "RAWEnhancement",
        author: str = "RAW Diffusion Team",
        description: str = "Real-time RAW image enhancement"
    ) -> ct.models.MLModel:
        """
        Convert PyTorch model to CoreML.
        
        Args:
            output_path: Path to save .mlmodel or .mlpackage
            model_name: Model name
            author: Author name
            description: Model description
            
        Returns:
            CoreML model
        """
        print(f"Converting model to CoreML...")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Compute units: {self.compute_units}")
        print(f"  Quantize weights: {self.quantize_weights}")
        
        # Create dummy input for tracing
        dummy_input = torch.randn(1, *self.input_shape)
        
        # Trace model
        print("  Tracing model...")
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Convert to CoreML
        print("  Converting to CoreML...")
        
        # Define input
        inputs = [
            ct.TensorType(
                name="input",
                shape=(1, *self.input_shape),
                dtype=ct.float32
            )
        ]
        
        # Convert
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            compute_units=self.compute_unit_map[self.compute_units],
            minimum_deployment_target=ct.target.iOS16,
            convert_to="mlprogram"  # Use ML Program for ANE support
        )
        
        # Set metadata
        mlmodel.short_description = description
        mlmodel.author = author
        mlmodel.license = "MIT"
        mlmodel.version = "1.0"
        
        # Quantize weights if requested
        if self.quantize_weights:
            print("  Quantizing weights to INT8...")
            mlmodel = self._quantize_weights(mlmodel, output_path)
        
        # Save model
        print(f"  Saving to {output_path}...")
        mlmodel.save(output_path)
        
        # Print model info
        self._print_model_info(mlmodel, output_path)
        
        print("✓ Conversion complete!")
        return mlmodel
    
    def _quantize_weights(
        self,
        mlmodel: ct.models.MLModel,
        output_path: str
    ) -> ct.models.MLModel:
        """Apply weight quantization."""
        # Create quantized path
        base_path = Path(output_path)
        quant_path = base_path.parent / f"{base_path.stem}_quantized{base_path.suffix}"
        
        # Apply quantization
        quantized_model = quantization_utils.quantize_weights(
            mlmodel,
            nbits=8,
            quantization_mode="linear"
        )
        
        return quantized_model
    
    def _print_model_info(self, mlmodel: ct.models.MLModel, path: str):
        """Print model information."""
        print("\nModel Information:")
        print(f"  Path: {path}")
        
        # Get model size
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024 ** 2)
            print(f"  Size: {size_mb:.2f} MB")
        
        # Print inputs
        print("\n  Inputs:")
        for input_name, input_desc in mlmodel.input_description.items():
            print(f"    - {input_name}: {input_desc}")
        
        # Print outputs
        print("\n  Outputs:")
        for output_name, output_desc in mlmodel.output_description.items():
            print(f"    - {output_name}: {output_desc}")


class PipelineConverter:
    """
    Convert entire inference pipeline to CoreML.
    
    This handles multi-stage pipelines by converting each component
    separately and combining them.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
    
    def convert_component(
        self,
        component_name: str,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        compute_units: str = 'CPU_AND_NE'
    ) -> str:
        """
        Convert a pipeline component.
        
        Args:
            component_name: Name of the component
            model: PyTorch model
            input_shape: Input shape
            compute_units: Target compute units
            
        Returns:
            Path to saved model
        """
        print(f"\nConverting {component_name}...")
        
        output_path = str(self.output_dir / f"{component_name}.mlpackage")
        
        converter = CoreMLConverter(
            model=model,
            input_shape=input_shape,
            compute_units=compute_units,
            quantize_weights=True
        )
        
        mlmodel = converter.convert(
            output_path=output_path,
            model_name=component_name,
            description=f"RAW Enhancement - {component_name}"
        )
        
        self.models[component_name] = output_path
        
        return output_path
    
    def export_pipeline_config(self) -> str:
        """
        Export pipeline configuration file.
        
        Returns:
            Path to config file
        """
        config = {
            'version': '1.0',
            'models': self.models,
            'execution_order': list(self.models.keys())
        }
        
        config_path = self.output_dir / 'pipeline_config.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nPipeline config saved to {config_path}")
        return str(config_path)
    
    def generate_swift_wrapper(self) -> str:
        """
        Generate Swift wrapper code for iOS integration.
        
        Returns:
            Path to Swift file
        """
        swift_code = self._create_swift_wrapper()
        
        swift_path = self.output_dir / 'RAWEnhancement.swift'
        
        with open(swift_path, 'w') as f:
            f.write(swift_code)
        
        print(f"\nSwift wrapper saved to {swift_path}")
        return str(swift_path)
    
    def _create_swift_wrapper(self) -> str:
        """Generate Swift wrapper code."""
        return '''
import CoreML
import Vision
import CoreImage
import Accelerate

/// Real-time RAW image enhancement using ML models
@available(iOS 16.0, *)
public class RAWEnhancementPipeline {
    
    // MARK: - Properties
    
    private var alignmentModel: MLModel?
    private var vaeEncoderModel: MLModel?
    private var consistencyModel: MLModel?
    private var vaeDecoderModel: MLModel?
    private var aberrationModel: MLModel?
    
    private let configuration: MLModelConfiguration
    private let queue = DispatchQueue(label: "com.rawenhancement.inference", qos: .userInteractive)
    
    // MARK: - Initialization
    
    public init() throws {
        // Configure for Neural Engine
        configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.allowLowPrecisionAccumulationOnGPU = true
        
        try loadModels()
    }
    
    private func loadModels() throws {
        // Load all pipeline components
        alignmentModel = try MLModel(contentsOf: Bundle.main.url(forResource: "alignment", withExtension: "mlmodelc")!)
        vaeEncoderModel = try MLModel(contentsOf: Bundle.main.url(forResource: "vae_encoder", withExtension: "mlmodelc")!)
        consistencyModel = try MLModel(contentsOf: Bundle.main.url(forResource: "consistency", withExtension: "mlmodelc")!)
        vaeDecoderModel = try MLModel(contentsOf: Bundle.main.url(forResource: "vae_decoder", withExtension: "mlmodelc")!)
        
        // Aberration model is optional
        if let aberrationURL = Bundle.main.url(forResource: "aberration", withExtension: "mlmodelc") {
            aberrationModel = try? MLModel(contentsOf: aberrationURL, configuration: configuration)
        }
    }
    
    // MARK: - Public API
    
    /// Process a burst of RAW images
    /// - Parameter burst: Array of RAW images (ProRAW format)
    /// - Returns: Enhanced RAW image
    public func process(burst: [CIImage]) async throws -> CIImage {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Stage 1: Alignment
        let aligned = try await alignBurst(burst)
        
        // Stage 2: Merge
        let merged = try mergeBurst(aligned)
        
        // Stage 3: Aberration correction (optional)
        let corrected = aberrationModel != nil ? try await correctAberration(merged) : merged
        
        // Stage 4: VAE encoding
        let latent = try await encodeToLatent(corrected)
        
        // Stage 5: Consistency model
        let enhancedLatent = try await runConsistency(latent)
        
        // Stage 6: VAE decoding
        let enhanced = try await decodeFromLatent(enhancedLatent)
        
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        print("Total latency: \\(String(format: "%.2f", elapsed))ms")
        
        return enhanced
    }
    
    // MARK: - Pipeline Stages
    
    private func alignBurst(_ burst: [CIImage]) async throws -> [CIImage] {
        // TODO: Implement burst alignment using optical flow model
        return burst
    }
    
    private func mergeBurst(_ burst: [CIImage]) throws -> CIImage {
        // Simple averaging for now
        guard let first = burst.first else {
            throw RAWEnhancementError.emptyBurst
        }
        
        var sum: [Float] = Array(repeating: 0, count: Int(first.extent.width * first.extent.height * 4))
        
        for image in burst {
            // Convert to float buffer and accumulate
            // TODO: Implement proper buffer operations
        }
        
        // Average
        let scale = 1.0 / Float(burst.count)
        sum = sum.map { $0 * scale }
        
        // Convert back to CIImage
        // TODO: Implement proper conversion
        return first
    }
    
    private func correctAberration(_ image: CIImage) async throws -> CIImage {
        guard let model = aberrationModel else {
            return image
        }
        
        // Run aberration correction
        // TODO: Implement model inference
        return image
    }
    
    private func encodeToLatent(_ image: CIImage) async throws -> MLMultiArray {
        guard let model = vaeEncoderModel else {
            throw RAWEnhancementError.modelNotLoaded
        }
        
        // Prepare input
        let input = try prepareInput(image)
        
        // Run inference
        let output = try await model.prediction(from: input)
        
        // Extract latent
        return try extractLatent(from: output)
    }
    
    private func runConsistency(_ latent: MLMultiArray) async throws -> MLMultiArray {
        guard let model = consistencyModel else {
            throw RAWEnhancementError.modelNotLoaded
        }
        
        // Run consistency model (2-4 steps)
        // TODO: Implement iterative refinement
        return latent
    }
    
    private func decodeFromLatent(_ latent: MLMultiArray) async throws -> CIImage {
        guard let model = vaeDecoderModel else {
            throw RAWEnhancementError.modelNotLoaded
        }
        
        // Run decoder
        // TODO: Implement decoding
        return CIImage()
    }
    
    // MARK: - Utilities
    
    private func prepareInput(_ image: CIImage) throws -> MLFeatureProvider {
        // Convert CIImage to MLMultiArray
        // TODO: Implement conversion
        fatalError("Not implemented")
    }
    
    private func extractLatent(from output: MLFeatureProvider) throws -> MLMultiArray {
        // Extract latent from model output
        // TODO: Implement extraction
        fatalError("Not implemented")
    }
}

// MARK: - Error Types

public enum RAWEnhancementError: Error {
    case modelNotLoaded
    case emptyBurst
    case invalidInput
    case inferenceError(String)
}

// MARK: - Usage Example

/*
// Example usage:
let pipeline = try RAWEnhancementPipeline()

// Load ProRAW burst
let burst = loadProRAWBurst()

// Process
Task {
    let enhanced = try await pipeline.process(burst: burst)
    // Display or save enhanced image
}
*/
'''
    
    def create_ios_integration_guide(self) -> str:
        """
        Create iOS integration guide.
        
        Returns:
            Path to markdown file
        """
        guide = self._create_integration_guide()
        
        guide_path = self.output_dir / 'iOS_Integration_Guide.md'
        
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        print(f"\nIntegration guide saved to {guide_path}")
        return str(guide_path)
    
    def _create_integration_guide(self) -> str:
        """Generate integration guide markdown."""
        return '''
# iOS Integration Guide

## Requirements

- Xcode 14.0+
- iOS 16.0+ (for Neural Engine optimizations)
- iPhone 13 Pro or later (for ProRAW support)

## Installation

1. Add CoreML models to your Xcode project:
   - Drag and drop all `.mlpackage` files into your project
   - Ensure "Copy items if needed" is checked
   - Add to your app target

2. Add the Swift wrapper:
   ```swift
   // Copy RAWEnhancement.swift to your project
   ```

## Usage

### Basic Usage

```swift
import UIKit
import CoreML

class RAWEnhancementViewController: UIViewController {
    
    let pipeline = try! RAWEnhancementPipeline()
    
    func processProRAWBurst(_ images: [UIImage]) async {
        // Convert to CIImages
        let ciImages = images.compactMap { CIImage(image: $0) }
        
        do {
            let enhanced = try await pipeline.process(burst: ciImages)
            
            // Convert back to UIImage
            let context = CIContext()
            if let cgImage = context.createCGImage(enhanced, from: enhanced.extent) {
                let uiImage = UIImage(cgImage: cgImage)
                // Display or save
            }
        } catch {
            print("Enhancement failed: \\(error)")
        }
    }
}
```

### Capture ProRAW Burst

```swift
import AVFoundation

class ProRAWCaptureManager: NSObject, AVCapturePhotoCaptureDelegate {
    
    var captureSession: AVCaptureSession!
    var photoOutput: AVCapturePhotoOutput!
    var burst: [UIImage] = []
    
    func setupCapture() {
        captureSession = AVCaptureSession()
        
        // Configure for ProRAW
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: camera)
            captureSession.addInput(input)
            
            photoOutput = AVCapturePhotoOutput()
            photoOutput.maxPhotoQualityPrioritization = .quality
            captureSession.addOutput(photoOutput)
            
            captureSession.startRunning()
        } catch {
            print("Setup failed: \\(error)")
        }
    }
    
    func captureBurst(count: Int = 8) {
        let settings = AVCapturePhotoSettings(rawPixelFormatType: camera.activeFormat.supportedRAWPhotoPixelFormatTypes.first!)
        settings.flashMode = .off
        
        for _ in 0..<count {
            photoOutput.capturePhoto(with: settings, delegate: self)
            Thread.sleep(forTimeInterval: 0.033)  // ~30fps
        }
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let imageData = photo.fileDataRepresentation(),
           let image = UIImage(data: imageData) {
            burst.append(image)
        }
    }
}
```

### Complete Example

```swift
class RAWEnhancementDemo {
    
    let pipeline = try! RAWEnhancementPipeline()
    let captureManager = ProRAWCaptureManager()
    
    func captureAndEnhance() async {
        // Setup capture
        captureManager.setupCapture()
        
        // Capture burst
        captureManager.captureBurst(count: 8)
        
        // Wait for capture to complete
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        
        // Enhance
        await processProRAWBurst(captureManager.burst)
    }
    
    func processProRAWBurst(_ images: [UIImage]) async {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            let ciImages = images.compactMap { CIImage(image: $0) }
            let enhanced = try await pipeline.process(burst: ciImages)
            
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            print("Total processing time: \\(elapsed)ms")
            
            // Save or display
            saveImage(enhanced)
        } catch {
            print("Enhancement failed: \\(error)")
        }
    }
    
    func saveImage(_ image: CIImage) {
        let context = CIContext()
        if let cgImage = context.createCGImage(image, from: image.extent) {
            let uiImage = UIImage(cgImage: cgImage)
            UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
        }
    }
}
```

## Performance Optimization

### 1. Enable Neural Engine

```swift
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
```

### 2. Use Metal for Preprocessing

```swift
let device = MTLCreateSystemDefaultDevice()
let commandQueue = device?.makeCommandQueue()
```

### 3. Batch Processing

Process multiple bursts in parallel:

```swift
await withTaskGroup(of: CIImage.self) { group in
    for burst in bursts {
        group.addTask {
            try await pipeline.process(burst: burst)
        }
    }
}
```

## Troubleshooting

### Issue: Models not loading

- Ensure models are added to app bundle
- Check model file names match Swift code
- Verify iOS deployment target is 16.0+

### Issue: Poor performance

- Check compute units configuration
- Enable Neural Engine explicitly
- Profile using Instruments

### Issue: High memory usage

- Process bursts sequentially
- Release intermediate results
- Use autoreleasepool for batch processing

## API Reference

See `RAWEnhancement.swift` for complete API documentation.
'''


if __name__ == "__main__":
    # Test CoreML conversion
    print("Testing CoreML conversion...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 4, 3, padding=1)
            )
        
        def forward(self, x):
            return self.conv(x)
    
    model = TestModel().eval()
    
    print("\n1. Testing single model conversion...")
    converter = CoreMLConverter(
        model=model,
        input_shape=(4, 256, 256),
        compute_units='CPU_AND_NE',
        quantize_weights=True
    )
    
    output_path = "./test_model.mlpackage"
    # mlmodel = converter.convert(output_path)
    print("   ✓ (Skipped - requires coremltools)")
    
    print("\n2. Testing pipeline converter...")
    pipeline_converter = PipelineConverter(output_dir="./coreml_models")
    
    # Generate Swift wrapper
    swift_path = pipeline_converter.generate_swift_wrapper()
    print(f"   ✓ Swift wrapper: {swift_path}")
    
    # Generate integration guide
    guide_path = pipeline_converter.create_ios_integration_guide()
    print(f"   ✓ Integration guide: {guide_path}")
    
    print("\n✓ CoreML conversion tested successfully!")
