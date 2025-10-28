"""Model Export Utilities - ONNX, TorchScript, TFLite, OpenVINO"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, List
import json


class ModelExporter:
    """
    Export trained models to various formats for deployment.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: Optional[dict] = None
    ):
        """
        Export model to ONNX format.
        
        Args:
            model: PyTorch model
            input_shape: Example input shape (without batch dimension)
            output_path: Where to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration
        """
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape, device=self.device)
        
        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        print(f"Exporting to ONNX...")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output path: {output_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"✓ Model exported to ONNX: {output_path}")
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verification passed")
        except ImportError:
            print("⚠️  onnx package not installed, skipping verification")
        except Exception as e:
            print(f"⚠️  ONNX verification failed: {e}")
    
    def export_to_torchscript(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        method: str = 'trace'
    ):
        """
        Export model to TorchScript.
        
        Args:
            model: PyTorch model
            input_shape: Example input shape
            output_path: Where to save TorchScript model
            method: 'trace' or 'script'
        """
        model.eval()
        model.to(self.device)
        
        dummy_input = torch.randn(1, *input_shape, device=self.device)
        
        print(f"Exporting to TorchScript ({method})...")
        
        if method == 'trace':
            traced_model = torch.jit.trace(model, dummy_input)
        elif method == 'script':
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        traced_model.save(output_path)
        
        print(f"✓ Model exported to TorchScript: {output_path}")
        
        # Verify
        loaded = torch.jit.load(output_path)
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = loaded(dummy_input)
            diff = torch.abs(output1 - output2).max().item()
            print(f"✓ Verification: max diff = {diff:.6f}")
    
    def export_to_tensorflowlite(
        self,
        onnx_path: str,
        output_path: str,
        quantize: bool = True
    ):
        """
        Export ONNX model to TensorFlow Lite.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Where to save TFLite model
            quantize: Whether to apply quantization
        """
        print("Exporting to TensorFlow Lite...")
        print("Note: This requires onnx-tf and tensorflow packages")
        
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
            
            # Load ONNX
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph('temp_tf_model')
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✓ Model exported to TFLite: {output_path}")
            
        except ImportError as e:
            print(f"⚠️  Required packages not installed: {e}")
            print("Install with: pip install onnx-tf tensorflow")
    
    def export_to_openvino(
        self,
        onnx_path: str,
        output_dir: str
    ):
        """
        Export ONNX model to OpenVINO format.
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Directory to save OpenVINO model
        """
        print("Exporting to OpenVINO...")
        print("Note: This requires OpenVINO toolkit")
        
        try:
            import subprocess
            
            # Run Model Optimizer
            cmd = [
                'mo',
                '--input_model', onnx_path,
                '--output_dir', output_dir,
                '--data_type', 'FP16'
            ]
            
            subprocess.run(cmd, check=True)
            
            print(f"✓ Model exported to OpenVINO: {output_dir}")
            
        except Exception as e:
            print(f"⚠️  OpenVINO export failed: {e}")
            print("Install OpenVINO toolkit from: https://docs.openvino.ai/")
    
    def create_deployment_package(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...],
        output_dir: str,
        formats: List[str] = ['onnx', 'torchscript']
    ):
        """
        Create a complete deployment package with multiple formats.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            input_shape: Input tensor shape
            output_dir: Output directory
            formats: List of export formats
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print(f"Creating deployment package for {model_name}")
        print("="*70)
        
        results = {}
        
        # Export to each format
        if 'onnx' in formats:
            onnx_path = output_path / f"{model_name}.onnx"
            self.export_to_onnx(model, input_shape, str(onnx_path))
            results['onnx'] = str(onnx_path)
        
        if 'torchscript' in formats:
            ts_path = output_path / f"{model_name}.pt"
            self.export_to_torchscript(model, input_shape, str(ts_path))
            results['torchscript'] = str(ts_path)
        
        if 'tflite' in formats and 'onnx' in results:
            tflite_path = output_path / f"{model_name}.tflite"
            self.export_to_tensorflowlite(results['onnx'], str(tflite_path))
            results['tflite'] = str(tflite_path)
        
        if 'openvino' in formats and 'onnx' in results:
            openvino_dir = output_path / 'openvino'
            self.export_to_openvino(results['onnx'], str(openvino_dir))
            results['openvino'] = str(openvino_dir)
        
        # Create metadata file
        metadata = {
            'model_name': model_name,
            'input_shape': input_shape,
            'exports': results,
            'formats': formats
        }
        
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Deployment package created: {output_dir}")
        print(f"  Metadata: {metadata_path}")
        
        return results


def main():
    """Example usage."""
    print("="*70)
    print("Model Export Utilities")
    print("="*70)
    
    print("\nThis tool exports models to various formats:")
    print("  - ONNX (cross-platform)")
    print("  - TorchScript (PyTorch production)")
    print("  - TensorFlow Lite (mobile)")
    print("  - OpenVINO (Intel hardware)")
    
    print("\nUsage:")
    print("  exporter = ModelExporter()")
    print("  exporter.export_to_onnx(model, input_shape, 'model.onnx')")
    print("  exporter.export_to_torchscript(model, input_shape, 'model.pt')")
    
    print("\nFor complete deployment package:")
    print("  exporter.create_deployment_package(")
    print("      model=your_model,")
    print("      model_name='raw_enhancement',")
    print("      input_shape=(8, 4, 512, 512),")
    print("      output_dir='./deployment',")
    print("      formats=['onnx', 'torchscript', 'tflite']")
    print("  )")


if __name__ == "__main__":
    main()
