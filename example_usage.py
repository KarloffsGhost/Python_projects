"""
Example usage of the modular LLM Hardware Checker
Shows how to use individual components programmatically
"""

from llm_hardware_checker import (
    HardwareDetector,
    ModelDatabase,
    CompatibilityChecker,
    PerformanceEstimator,
    DisplayManager
)


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===\n")
    
    # Detect hardware
    detector = HardwareDetector()
    hardware = detector.detect_hardware()
    
    print(f"Detected GPU: {hardware.gpu_model}")
    print(f"VRAM: {hardware.gpu_vram_gb}GB")
    print(f"RAM: {hardware.system_ram_gb}GB")
    

def example_check_specific_model():
    """Check if a specific model can run"""
    print("\n=== Check Specific Model ===\n")
    
    # Detect hardware
    detector = HardwareDetector()
    hardware = detector.detect_hardware()
    
    # Search for model
    model_db = ModelDatabase()
    model_info = model_db.search_model_info("Mistral 7B")
    
    if model_info:
        print(f"Found model: {model_info.name}")
        print(f"Parameters: {model_info.parameter_count}")
        
        # Check compatibility
        checker = CompatibilityChecker()
        results = checker.check_compatibility(hardware, model_info)
        
        # Get compatible options
        compatible = checker.get_compatible_options(results)
        print(f"\nCompatible quantizations: {len(compatible)}")
        
        for quant, result in list(compatible.items())[:3]:
            print(f"  - {quant}: {result.size_gb:.1f}GB, "
                  f"{result.performance_metrics.tokens_per_second:.1f} tok/s")


def example_performance_comparison():
    """Compare performance of different models"""
    print("\n=== Performance Comparison ===\n")
    
    # Detect hardware
    detector = HardwareDetector()
    hardware = detector.detect_hardware()
    
    # Models to compare
    models = ["Phi-2", "Mistral 7B", "Llama 2 13B"]
    model_db = ModelDatabase()
    
    print(f"Hardware: {hardware.gpu_model} ({hardware.gpu_vram_gb}GB VRAM)\n")
    
    for model_name in models:
        model_info = model_db.search_model_info(model_name)
        if model_info:
            # Check Q4_K_M quantization
            if "Q4_K_M" in model_info.quantization_info:
                quant_info = model_info.quantization_info["Q4_K_M"]
                
                # Estimate performance
                metrics = PerformanceEstimator.estimate_performance(
                    hardware,
                    quant_info["size_gb"],
                    "Q4_K_M",
                    model_info.parameter_count
                )
                
                print(f"{model_name}:")
                print(f"  Size: {quant_info['size_gb']:.1f}GB")
                print(f"  Speed: {metrics.tokens_per_second:.1f} tokens/sec")
                print(f"  Quality: {quant_info['quality']}")
                print()


def example_find_best_model():
    """Find the best model for your hardware"""
    print("\n=== Find Best Model for Hardware ===\n")
    
    # Detect hardware
    detector = HardwareDetector()
    hardware = detector.detect_hardware()
    
    # Get recommendations
    model_db = ModelDatabase()
    
    if hardware.has_gpu:
        recommendations = model_db.get_model_recommendations(
            hardware.gpu_vram_gb, 
            use_case="general"
        )
    else:
        recommendations = model_db.get_model_recommendations(
            0,  # No GPU
            use_case="general"
        )
    
    print(f"Recommended models for your hardware:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec}")


def example_custom_model_check():
    """Check a custom model not in the database"""
    print("\n=== Custom Model Check ===\n")
    
    # This would check a model that might not be in our database
    # The system will try to find it on HuggingFace and infer capabilities
    
    detector = HardwareDetector()
    hardware = detector.detect_hardware()
    
    model_db = ModelDatabase()
    model_info = model_db.search_model_info("StableLM 3B")  # Example custom model
    
    if model_info:
        print(f"Found model: {model_info.name}")
        if model_info.capabilities:
            print(f"Inferred strength: {model_info.capabilities.primary_strength}")
        
        # Quick compatibility check
        checker = CompatibilityChecker()
        results = checker.check_compatibility(hardware, model_info)
        compatible = checker.get_compatible_options(results)
        
        if compatible:
            print(f"✅ This model can run on your hardware!")
            print(f"   Compatible quantizations: {len(compatible)}")
        else:
            print(f"❌ This model cannot run on your hardware")


def example_programmatic_integration():
    """Example of integrating into another application"""
    print("\n=== Programmatic Integration Example ===\n")
    
    class MyLLMApp:
        def __init__(self):
            self.hardware_detector = HardwareDetector()
            self.model_db = ModelDatabase()
            self.checker = CompatibilityChecker()
            
            # Detect hardware once at startup
            self.hardware = self.hardware_detector.detect_hardware()
            
        def can_run_model(self, model_name: str) -> bool:
            """Check if a model can run on this system"""
            model_info = self.model_db.search_model_info(model_name)
            if not model_info:
                return False
            
            results = self.checker.check_compatibility(self.hardware, model_info)
            compatible = self.checker.get_compatible_options(results)
            
            return len(compatible) > 0
        
        def get_best_quantization(self, model_name: str) -> str:
            """Get the best quantization for a model"""
            model_info = self.model_db.search_model_info(model_name)
            if not model_info:
                return None
            
            results = self.checker.check_compatibility(self.hardware, model_info)
            best_options = self.checker.get_best_options(results)
            
            if 'balanced' in best_options:
                return best_options['balanced'][0]  # Return quantization name
            
            return None
    
    # Use the app
    app = MyLLMApp()
    
    # Check models
    models_to_check = ["Phi-2", "Llama 2 70B", "Mistral 7B"]
    
    for model in models_to_check:
        can_run = app.can_run_model(model)
        if can_run:
            best_quant = app.get_best_quantization(model)
            print(f"✅ {model}: Use {best_quant}")
        else:
            print(f"❌ {model}: Cannot run")


if __name__ == "__main__":
    print("LLM Hardware Checker - Example Usage\n")
    print("This demonstrates various ways to use the modular components\n")
    
    # Run examples
    example_basic_usage()
    example_check_specific_model()
    example_performance_comparison()
    example_find_best_model()
    example_custom_model_check()
    example_programmatic_integration()
    
    print("\n✨ Examples completed!")