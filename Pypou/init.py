from Pypou.dynamic_api import DynamicAPI
from Pypou.exceptions import APIError, InvalidConfigError, AuthenticationError

def get_api(config_name: str, **auth_kwargs) -> DynamicAPI:
    """Charge une API dynamiquement à partir de son nom."""
    return DynamicAPI(config_name, **auth_kwargs)

__all__ = ["get_api", "APIError", "InvalidConfigError", "AuthenticationError"]
