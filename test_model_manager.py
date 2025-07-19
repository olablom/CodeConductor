#!/usr/bin/env python3
"""
Test script for ModelManager
"""

from integrations.model_manager import ModelManager


def test_model_manager():
    """Test the ModelManager functionality"""
    print("🧪 Testing ModelManager...")

    try:
        # Create model manager
        model_manager = ModelManager()
        print("✅ ModelManager created successfully")

        # Get model information
        model_info = model_manager.get_model_info()
        print(f"📊 Model Information:")
        print(f"  Server URL: {model_info['server_url']}")
        print(f"  Default model: {model_info['default_model']}")
        print(f"  Configured models: {len(model_info['configured_models'])}")
        print(
            f"  Available for installation: {len(model_info['available_for_installation'])}"
        )
        print(f"  Currently loaded: {len(model_info['currently_loaded'])}")
        print(f"  LM Studio CLI available: {model_info['lm_studio_cli_available']}")

        # List available models
        print(f"\n🔍 Available models for installation:")
        available_models = model_manager.list_available_models()
        for i, model in enumerate(available_models[:10], 1):  # Show first 10
            print(f"  {i}. {model}")
        if len(available_models) > 10:
            print(f"  ... and {len(available_models) - 10} more")

        # List local models
        print(f"\n📦 Currently loaded models:")
        local_models = model_manager.list_local_models()
        for i, model in enumerate(local_models, 1):
            print(f"  {i}. {model}")

        if not local_models:
            print("  No models currently loaded")

        # Test model checking
        print(f"\n🔧 Testing model availability:")
        if model_info["configured_models"]:
            test_model = model_info["configured_models"][0]
            is_installed = model_manager.is_model_installed(test_model)
            is_loaded = model_manager.is_model_loaded(test_model)
            print(f"  Test model: {test_model}")
            print(f"  Available for installation: {'✅' if is_installed else '❌'}")
            print(f"  Currently loaded: {'✅' if is_loaded else '❌'}")
        else:
            print("  No configured models to test")

        # Test recommended models
        print(f"\n💡 Recommended models:")
        recommended = model_manager.recommend_models()
        for i, model in enumerate(recommended, 1):
            print(f"  {i}. {model}")

        if not recommended:
            print("  No recommended models available")

        return True

    except Exception as e:
        print(f"❌ ModelManager test failed: {e}")
        return False


def test_model_setup():
    """Test setting up default models"""
    print("\n🧪 Testing model setup...")

    try:
        model_manager = ModelManager()

        # Test setup default models
        success = model_manager.setup_default_models()

        if success:
            print("✅ Default models setup completed")

            # Check updated config
            model_info = model_manager.get_model_info()
            print(f"📊 Updated configuration:")
            print(f"  Configured models: {len(model_info['configured_models'])}")
            print(f"  Default model: {model_info['default_model']}")
        else:
            print("❌ Default models setup failed")

        return success

    except Exception as e:
        print(f"❌ Model setup test failed: {e}")
        return False


def test_config_update():
    """Test updating configuration"""
    print("\n🧪 Testing configuration update...")

    try:
        model_manager = ModelManager()

        # Test models
        test_models = [
            "meta-llama/Llama-3-7b",
            "hf://coder-llama/CodeLlama-7b-instruct",
            "gpt2",
        ]

        # Update configuration
        success = model_manager.update_config(
            models=test_models, default_model="meta-llama/Llama-3-7b"
        )

        if success:
            print("✅ Configuration updated successfully")

            # Verify update
            model_info = model_manager.get_model_info()
            print(f"📊 Updated configuration:")
            print(f"  Configured models: {model_info['configured_models']}")
            print(f"  Default model: {model_info['default_model']}")
        else:
            print("❌ Configuration update failed")

        return success

    except Exception as e:
        print(f"❌ Configuration update test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎼 CodeConductor v2.0 - ModelManager Test")
    print("=" * 60)

    success = True
    success &= test_model_manager()
    success &= test_model_setup()
    success &= test_config_update()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All ModelManager tests passed!")
        print("🔧 Your model management system is working!")
    else:
        print("❌ Some ModelManager tests failed. Check the errors above.")

    print("=" * 60)
