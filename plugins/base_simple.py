"""
Simplified Plugin Architecture for CodeConductor v2.0

Based on user's elegant plugin design, adapted for CodeConductor integration.
"""

import importlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


class BasePlugin(ABC):
    """Base class for all CodeConductor plugins"""

    @abstractmethod
    def name(self) -> str:
        """Unique plugin name"""
        pass

    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass

    @abstractmethod
    def description(self) -> str:
        """Plugin description"""
        pass

    @abstractmethod
    def activate(self) -> None:
        """Activation hook"""
        pass

    @abstractmethod
    def deactivate(self) -> None:
        """Deactivation hook"""
        pass


class BaseAgentPlugin(BasePlugin):
    """Base class for agent plugins that can analyze code"""

    @abstractmethod
    def analyze(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis and return results"""
        pass


class BaseToolPlugin(BasePlugin):
    """Base class for tool plugins that provide utility functions"""

    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> Any:
        """Execute the tool with input data"""
        pass


class PluginManager:
    """
    Discovers and manages plugins under a given namespace or folder.
    Adapted for CodeConductor integration.
    """

    def __init__(self, namespace: str = "plugins"):
        self.namespace = namespace
        self.plugins: Dict[str, BasePlugin] = {}
        self.logger = logging.getLogger("PluginManager")
        self.config_file = Path("config/plugins_config.yaml")

        # Load configuration
        self.load_config()

    def discover(self) -> None:
        """
        Discover all plugin modules in the namespace and instantiate them.
        """
        try:
            # Ensure plugins directory exists
            plugins_dir = Path(self.namespace)
            if not plugins_dir.exists():
                plugins_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created plugins directory: {plugins_dir}")
                return

            # Discover Python modules
            for py_file in plugins_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                try:
                    module_name = f"{self.namespace}.{py_file.stem}"
                    module = importlib.import_module(module_name)

                    # Find plugin classes in module
                    for attr in dir(module):
                        cls = getattr(module, attr)
                        if (
                            isinstance(cls, type)
                            and issubclass(cls, BasePlugin)
                            and cls is not BasePlugin
                            and cls is not BaseAgentPlugin
                            and cls is not BaseToolPlugin
                        ):
                            # Check if plugin is enabled in config
                            try:
                                instance: BasePlugin = cls()
                                plugin_name = instance.name()
                                if self.is_plugin_enabled(plugin_name):
                                    self.plugins[plugin_name] = instance
                                    self.logger.info(
                                        f"Discovered plugin: {plugin_name}"
                                    )
                                else:
                                    self.logger.info(f"Plugin disabled: {plugin_name}")
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to instantiate plugin class {cls.__name__}: {e}"
                                )

                except Exception as e:
                    self.logger.error(f"Failed to load plugin from {py_file}: {e}")

        except Exception as e:
            self.logger.error(f"Plugin discovery failed: {e}")

    def activate_all(self) -> None:
        """Activate all discovered plugins"""
        for name, plugin in self.plugins.items():
            try:
                plugin.activate()
                self.logger.info(f"Activated plugin: {name}")
            except Exception as e:
                self.logger.error(f"Failed to activate plugin {name}: {e}")

    def deactivate_all(self) -> None:
        """Deactivate all plugins"""
        for name, plugin in self.plugins.items():
            try:
                plugin.deactivate()
                self.logger.info(f"Deactivated plugin: {name}")
            except Exception as e:
                self.logger.error(f"Failed to deactivate plugin {name}: {e}")

    def get_plugins(self) -> List[BasePlugin]:
        """Get all plugins"""
        return list(self.plugins.values())

    def get_agent_plugins(self) -> List[BaseAgentPlugin]:
        """Get only agent plugins"""
        return [p for p in self.plugins.values() if isinstance(p, BaseAgentPlugin)]

    def get_tool_plugins(self) -> List[BaseToolPlugin]:
        """Get only tool plugins"""
        return [p for p in self.plugins.values() if isinstance(p, BaseToolPlugin)]

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a specific plugin by name"""
        return self.plugins.get(name)

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin"""
        if name in self.plugins:
            self.config["enabled_plugins"].add(name)
            self.save_config()
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin"""
        if name in self.plugins:
            self.config["enabled_plugins"].discard(name)
            self.save_config()
            return True
        return False

    def is_plugin_enabled(self, name: str) -> bool:
        """Check if a plugin is enabled"""
        return name in self.config.get("enabled_plugins", set())

    def load_config(self) -> None:
        """Load plugin configuration"""
        self.config = {"enabled_plugins": set(), "plugin_settings": {}}

        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self.config["enabled_plugins"] = set(
                        data.get("enabled_plugins", [])
                    )
                    self.config["plugin_settings"] = data.get("plugin_settings", {})
            except Exception as e:
                self.logger.error(f"Failed to load plugin config: {e}")

    def save_config(self) -> None:
        """Save plugin configuration"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    {
                        "enabled_plugins": list(self.config["enabled_plugins"]),
                        "plugin_settings": self.config["plugin_settings"],
                    },
                    f,
                    default_flow_style=False,
                    indent=2,
                )
        except Exception as e:
            self.logger.error(f"Failed to save plugin config: {e}")

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about all plugins"""
        info = {
            "total_plugins": len(self.plugins),
            "agent_plugins": len(self.get_agent_plugins()),
            "tool_plugins": len(self.get_tool_plugins()),
            "enabled_plugins": len(self.config["enabled_plugins"]),
            "plugins": {},
        }

        for name, plugin in self.plugins.items():
            info["plugins"][name] = {
                "name": plugin.name(),
                "version": plugin.version(),
                "description": plugin.description(),
                "type": "agent" if isinstance(plugin, BaseAgentPlugin) else "tool",
                "enabled": self.is_plugin_enabled(name),
            }

        return info
