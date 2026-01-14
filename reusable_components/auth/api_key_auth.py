"""API Key Authentication Service."""
import hashlib
import logging
import os
from functools import wraps
from flask import request, jsonify
from google.cloud import secretmanager

logger = logging.getLogger(__name__)

class APIKeyAuth:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        self._api_keys = None
    
    def load_api_keys(self) -> dict:
        """Load API keys from Secret Manager."""
        if self._api_keys:
            return self._api_keys
        
        try:
            name = f"projects/{self.project_id}/secrets/API_KEYS/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            import json
            self._api_keys = json.loads(response.payload.data.decode('UTF-8'))
            return self._api_keys
        except Exception as e:
            logger.warning(f"Failed to load API keys: {e}")
            return {}
    
    def validate_key(self, api_key: str) -> bool:
        """Validate API key."""
        keys = self.load_api_keys()
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return key_hash in keys.values()
    
    def require_api_key(self, f):
        """Decorator to require API key."""
        @wraps(f)
        def decorated(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            if not api_key or not self.validate_key(api_key):
                return jsonify({"error": "Invalid API key"}), 401
            return f(*args, **kwargs)
        return decorated