"""
Plugin Architecture for CodeConductor v2.0

This module provides the foundation for a dynamic plugin system that allows
third-party developers to extend CodeConductor with custom agents, tools, and integrations.
"""

import abc
import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml


class PluginType(Enum):
    """Types of plugins supported by CodeConductor"""

    AGENT = "agent"
    TOOL = "tool"
    INTEGRATION = "integration"
    VALIDATOR = "validator"
    GENERATOR = "generator"


class PluginStatus(Enum):
    """Plugin status enumeration"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    license: Optional[str] = None


@dataclass
class PluginInfo:
    """Complete plugin information"""

    metadata: PluginMetadata
    status: PluginStatus
    error_message: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    is_enabled: bool = True


class BasePlugin(abc.ABC):
    """
    Abstract base class for all CodeConductor plugins.

    All plugins must inherit from this class and implement the required methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with configuration"""
        self.config = config or {}
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._validate_config()

    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources"""
        pass

    def _validate_config(self) -> None:
        """Validate plugin configuration against schema"""
        if self.metadata.config_schema:
            # Basic validation - could be enhanced with JSON Schema
            required_fields = [
                field
                for field, spec in self.metadata.config_schema.items()
                if spec.get("required", False)
            ]

            for field in required_fields:
                if field not in self.config:
                    raise ValueError(f"Required config field '{field}' not provided")

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value

    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)

    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)


class BaseAgentPlugin(BasePlugin):
    """
    Base class for agent plugins.

    Agent plugins extend the multi-agent system with custom reasoning and code generation capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.agent_type = "custom"

    @abc.abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current context and provide insights.

        Args:
            context: Current pipeline context including prompt, discussion history, etc.

        Returns:
            Analysis results as a dictionary
        """
        pass

    @abc.abstractmethod
    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take action based on analysis.

        Args:
            context: Current pipeline context

        Returns:
            Action results as a dictionary
        """
        pass

    @abc.abstractmethod
    def observe(self, result: Dict[str, Any]) -> None:
        """
        Observe the results of actions and learn.

        Args:
            result: Results from previous actions
        """
        pass


class BaseToolPlugin(BasePlugin):
    """
    Base class for tool plugins.

    Tool plugins provide utility functions that can be used by agents or the pipeline.
    """

    @abc.abstractmethod
    def execute(self, input_data: Any, **kwargs) -> Any:
        """
        Execute the tool with input data.

        Args:
            input_data: Input data for the tool
            **kwargs: Additional parameters

        Returns:
            Tool execution results
        """
        pass


class BaseIntegrationPlugin(BasePlugin):
    """
    Base class for integration plugins.

    Integration plugins connect CodeConductor to external services and APIs.
    """

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish connection to external service. Return True if successful."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from external service"""
        pass

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to external service"""
        pass


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Handles dynamic loading of plugins from various sources including
    local directories, PyPI packages, and custom repositories.
    """

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.plugin_dirs = plugin_dirs or ["plugins", "config/plugins"]
        self.plugins: Dict[str, PluginInfo] = {}
        self.logger = logging.getLogger("PluginManager")

        # Create plugin directories if they don't exist
        for plugin_dir in self.plugin_dirs:
            Path(plugin_dir).mkdir(parents=True, exist_ok=True)

    def discover_plugins(self) -> List[PluginInfo]:
        """Discover all available plugins"""
        discovered_plugins = []

        # Discover from plugin directories
        for plugin_dir in self.plugin_dirs:
            discovered_plugins.extend(self._discover_from_directory(plugin_dir))

        # Discover from entry points (if setuptools is available)
        try:
            discovered_plugins.extend(self._discover_from_entry_points())
        except ImportError:
            self.logger.info("setuptools not available, skipping entry point discovery")

        return discovered_plugins

    def _discover_from_directory(self, plugin_dir: str) -> List[PluginInfo]:
        """Discover plugins from a directory"""
        plugins = []
        plugin_path = Path(plugin_dir)

        if not plugin_path.exists():
            return plugins

        # Look for plugin configuration files
        for config_file in plugin_path.glob("*.yaml"):
            try:
                plugin_info = self._load_plugin_from_config(config_file)
                if plugin_info:
                    plugins.append(plugin_info)
            except Exception as e:
                self.logger.error(f"Failed to load plugin from {config_file}: {e}")

        # Look for Python modules
        for py_file in plugin_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                plugin_info = self._load_plugin_from_module(py_file)
                if plugin_info:
                    plugins.append(plugin_info)
            except Exception as e:
                self.logger.error(f"Failed to load plugin from {py_file}: {e}")

        return plugins

    def _discover_from_entry_points(self) -> List[PluginInfo]:
        """Discover plugins from setuptools entry points"""
        plugins = []

        try:
            import pkg_resources

            for entry_point in pkg_resources.iter_entry_points("codeconductor.plugins"):
                try:
                    plugin_class = entry_point.load()
                    plugin_instance = plugin_class()
                    plugin_info = PluginInfo(
                        metadata=plugin_instance.metadata,
                        status=PluginStatus.INACTIVE,
                        is_enabled=True,
                    )
                    plugins.append(plugin_info)
                except Exception as e:
                    self.logger.error(f"Failed to load entry point {entry_point}: {e}")

        except ImportError:
            pass

        return plugins

    def _load_plugin_from_config(self, config_file: Path) -> Optional[PluginInfo]:
        """Load plugin from YAML configuration file"""
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        metadata = PluginMetadata(
            name=config_data["name"],
            version=config_data["version"],
            description=config_data.get("description", ""),
            author=config_data.get("author", ""),
            plugin_type=PluginType(config_data["type"]),
            entry_point=config_data["entry_point"],
            dependencies=config_data.get("dependencies", []),
            config_schema=config_data.get("config_schema"),
            tags=config_data.get("tags", []),
            homepage=config_data.get("homepage"),
            license=config_data.get("license"),
        )

        return PluginInfo(
            metadata=metadata,
            status=PluginStatus.INACTIVE,
            config=config_data.get("config", {}),
            is_enabled=config_data.get("enabled", True),
        )

    def _load_plugin_from_module(self, module_file: Path) -> Optional[PluginInfo]:
        """Load plugin from Python module"""
        # Import the module
        module_name = module_file.stem
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find plugin classes
        plugin_classes = []
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BasePlugin)
                and obj != BasePlugin
                and obj != BaseAgentPlugin
                and obj != BaseToolPlugin
                and obj != BaseIntegrationPlugin
            ):
                plugin_classes.append(obj)

        if not plugin_classes:
            return None

        # Use the first plugin class found
        plugin_class = plugin_classes[0]

        # Create plugin instance with empty config to get metadata
        try:
            plugin_instance = plugin_class({})
            return PluginInfo(
                metadata=plugin_instance.metadata,
                status=PluginStatus.INACTIVE,
                is_enabled=True,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to create plugin instance for {module_name}: {e}"
            )
            return None

    def load_plugin(self, plugin_info: PluginInfo) -> Optional[BasePlugin]:
        """Load a specific plugin"""
        try:
            plugin_info.status = PluginStatus.LOADING

            # Import the plugin module
            module_name, class_name = plugin_info.metadata.entry_point.rsplit(":", 1)
            module = importlib.import_module(module_name)
            plugin_class = getattr(module, class_name)

            # Create plugin instance
            plugin_instance = plugin_class(plugin_info.config)

            # Initialize plugin
            if plugin_instance.initialize():
                plugin_info.status = PluginStatus.ACTIVE
                self.plugins[plugin_info.metadata.name] = plugin_info
                self.logger.info(
                    f"Successfully loaded plugin: {plugin_info.metadata.name}"
                )
                return plugin_instance
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Plugin initialization failed"
                self.logger.error(
                    f"Failed to initialize plugin: {plugin_info.metadata.name}"
                )
                return None

        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = str(e)
            self.logger.error(f"Failed to load plugin {plugin_info.metadata.name}: {e}")
            return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            return False

        plugin_info = self.plugins[plugin_name]
        try:
            # Plugin cleanup would happen here if we stored instances
            plugin_info.status = PluginStatus.INACTIVE
            del self.plugins[plugin_name]
            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information"""
        return self.plugins.get(plugin_name)

    def list_plugins(self) -> List[PluginInfo]:
        """List all plugins"""
        return list(self.plugins.values())

    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        plugin_info = self.get_plugin(plugin_name)
        if plugin_info:
            plugin_info.is_enabled = True
            return True
        return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        plugin_info = self.get_plugin(plugin_name)
        if plugin_info:
            plugin_info.is_enabled = False
            return True
        return False

    def save_plugin_config(self) -> None:
        """Save plugin configuration to file"""
        config_data = {
            "plugins": {
                name: {"enabled": info.is_enabled, "config": info.config}
                for name, info in self.plugins.items()
            }
        }

        config_file = Path("config/plugins_config.yaml")
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

    def load_plugin_config(self) -> None:
        """Load plugin configuration from file"""
        config_file = Path("config/plugins_config.yaml")
        if not config_file.exists():
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            for plugin_name, plugin_config in config_data.get("plugins", {}).items():
                plugin_info = self.get_plugin(plugin_name)
                if plugin_info:
                    plugin_info.is_enabled = plugin_config.get("enabled", True)
                    plugin_info.config.update(plugin_config.get("config", {}))

        except Exception as e:
            self.logger.error(f"Failed to load plugin config: {e}")


# Global plugin manager instance
plugin_manager = PluginManager()
