"""
Base registry class for dynamic module registration
"""
from typing import Any, Dict, List, Callable, Union, Optional
import importlib


class Registry:
    """
    A registry for storing and retrieving functions or classes by name.
    
    This class provides a decorator-based registration system that allows
    modules to register themselves with a string key for later retrieval.
    
    Example:
        >>> # Create a registry
        >>> REWARD_REGISTRY = Registry("rewards")
        >>> 
        >>> # Register a function
        >>> @REWARD_REGISTRY.register("my_reward")
        >>> def my_reward_func(pred, gold):
        >>>     return 1.0
        >>> 
        >>> # Retrieve the function
        >>> func = REWARD_REGISTRY.get("my_reward")
        >>> result = func("prediction", "gold")
    """
    
    def __init__(self, name: str, module_path: Optional[str] = None):
        """
        Initialize a new registry.
        
        Args:
            name: Name of this registry (used in error messages)
            module_path: Optional module path for lazy loading (e.g., "rlyx.rewards")
        """
        self._name = name
        self._module_path = module_path
        self._registry: Dict[str, Any] = {}
        self._loaded = False
    
    def register(self, key: str) -> Callable:
        """
        Decorator for registering a function or class.
        
        Args:
            key: String key to register the function/class under
            
        Returns:
            Decorator function
            
        Raises:
            KeyError: If the key is already registered
        """
        def decorator(fn_or_class: Union[Callable, type]) -> Union[Callable, type]:
            if key in self._registry:
                raise KeyError(
                    f"{key} is already registered in {self._name} registry. "
                    f"Existing: {self._registry[key]}, New: {fn_or_class}"
                )
            self._registry[key] = fn_or_class
            # Add the key as an attribute for debugging
            setattr(fn_or_class, '_registry_key', key)
            return fn_or_class
        return decorator
    
    def _lazy_load(self):
        """Lazy load all modules in the registry's module path."""
        if self._loaded or not self._module_path:
            return
        
        try:
            import os
            import pkgutil
            
            # Import the parent module
            parent_module = importlib.import_module(self._module_path)
            parent_path = parent_module.__path__[0]
            
            # Auto-discover and import all Python modules in the directory
            for filename in os.listdir(parent_path):
                if filename.endswith('.py') and filename not in ['__init__.py', 'registry.py']:
                    module_name = filename[:-3]  # Remove .py extension
                    try:
                        importlib.import_module(f"{self._module_path}.{module_name}")
                    except ImportError:
                        # Skip modules that can't be imported
                        pass
            
            self._loaded = True
        except (ImportError, AttributeError, FileNotFoundError):
            # If lazy loading fails, continue without it
            pass
    
    def get(self, key: str) -> Any:
        """
        Retrieve a registered function or class.
        
        Args:
            key: String key the function/class was registered under
            
        Returns:
            The registered function or class
            
        Raises:
            KeyError: If the key is not registered
        """
        # Try lazy loading if not already loaded
        self._lazy_load()
        
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{key} is not registered in {self._name} registry. "
                f"Available: [{available}]"
            )
        return self._registry[key]
    
    def list(self) -> List[str]:
        """
        List all registered keys.
        
        Returns:
            Sorted list of registered keys
        """
        # Try lazy loading if not already loaded
        self._lazy_load()
        return sorted(self._registry.keys())
    
    def __contains__(self, key: str) -> bool:
        """Check if a key is registered."""
        # Try lazy loading if not already loaded
        self._lazy_load()
        return key in self._registry
    
    def __len__(self) -> int:
        """Return the number of registered items."""
        return len(self._registry)
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"Registry(name={self._name}, items={self.list()})"