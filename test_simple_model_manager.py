#!/usr/bin/env python3
"""
Test script for SimpleModelManager
"""

from integrations.model_manager_simple import SimpleModelManager


def test_simple_model_manager():
    """Test the SimpleModelManager functionality"""
    print("🧪 Testing SimpleModelManager...")

    try:
        # Create model manager
        model_manager = SimpleModelManager()
        print("✅ SimpleModelManager created successfully")

        # Get model information
        model_info = model_manager.get_model_info()
        print("📊 Model Information:")
        print(f"  Server URL: {model_info['server_url']}")
        print(f"  Default model: {model_info['default_model']}")
        print(f"  Configured models: {len(model_info['configured_models'])}")
        print(f"  Downloaded models: {len(model_info['downloaded_models'])}")
        print(f"  Currently loaded: {len(model_info['currently_loaded'])}")
        print(f"  LM Studio CLI available: {model_info['lm_studio_cli_available']}")

        # List downloaded models
        print("\n📦 Downloaded models:")
        downloaded_models = model_manager.list_downloaded_models()
        for i, model in enumerate(downloaded_models, 1):
            print(f"  {i}. {model}")

        if not downloaded_models:
            print("  No models downloaded")

        # List local models
        print("\n🚀 Currently loaded models:")
        local_models = model_manager.list_local_models()
        for i, model in enumerate(local_models, 1):
            print(f"  {i}. {model}")

        if not local_models:
            print("  No models currently loaded")

        # Test model checking
        print("\n🔧 Testing model availability:")
        if model_info["configured_models"]:
            test_model = model_info["configured_models"][0]
            is_downloaded = model_manager.is_model_downloaded(test_model)
            is_loaded = model_manager.is_model_loaded(test_model)
            print(f"  Test model: {test_model}")
            print(f"  Downloaded: {'✅' if is_downloaded else '❌'}")
            print(f"  Currently loaded: {'✅' if is_loaded else '❌'}")
        else:
            print("  No configured models to test")

        # Test recommended models
        print("\n💡 Recommended models (from downloaded):")
        recommended = model_manager.recommend_models()
        for i, model in enumerate(recommended, 1):
            print(f"  {i}. {model}")

        if not recommended:
            print("  No recommended models downloaded")

        return True

    except Exception as e:
        print(f"❌ SimpleModelManager test failed: {e}")
        return False


def test_model_setup():
    """Test setting up default models"""
    print("\n🧪 Testing model setup...")

    try:
        model_manager = SimpleModelManager()

        # Test setup default models
        success = model_manager.setup_default_models()

        if success:
            print("✅ Default models setup completed")

            # Check updated config
            model_info = model_manager.get_model_info()
            print("📊 Updated configuration:")
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
        model_manager = SimpleModelManager()

        # Get downloaded models to use as test models
        downloaded_models = model_manager.list_downloaded_models()

        if downloaded_models:
            # Use first downloaded model as test
            test_models = [downloaded_models[0]]

            # Update configuration
            success = model_manager.update_config(
                models=test_models, default_model=test_models[0]
            )

            if success:
                print("✅ Configuration updated successfully")

                # Verify update
                model_info = model_manager.get_model_info()
                print("📊 Updated configuration:")
                print(f"  Configured models: {model_info['configured_models']}")
                print(f"  Default model: {model_info['default_model']}")
            else:
                print("❌ Configuration update failed")
        else:
            print("⚠️ No downloaded models available for testing")
            success = True  # Not a failure, just no models to test with

        return success

    except Exception as e:
        print(f"❌ Configuration update test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎼 CodeConductor v2.0 - SimpleModelManager Test")
    print("=" * 60)

    success = True
    success &= test_simple_model_manager()
    success &= test_model_setup()
    success &= test_config_update()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All SimpleModelManager tests passed!")
        print("🔧 Your simplified model management system is working!")
    else:
        print("❌ Some SimpleModelManager tests failed. Check the errors above.")

    print("=" * 60)
