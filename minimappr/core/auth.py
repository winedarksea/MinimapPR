"""
minimappr/core/auth.py
Authentication utilities for federated endpoints.
"""
import secrets
from typing import Optional

from fastapi.security.utils import get_authorization_scheme_param


def verify_constant_time(
    provided_token: str | None,
    expected_token: str | None,
) -> bool:
    """
    Verify a provided token against an expected secret using constant-time comparison.
    
    Design Principles:
    1. Secure: Uses secrets.compare_digest() to prevent timing analysis attacks.
    2. Flexible: If expected_token is None/empty, auth is disabled (returns True).
    3. Robust: Handles None inputs gracefully.
    """
    # If no secret is configured, we assume authentication is disabled (Low-Risk Mode)
    if not expected_token:
        return True
    
    # If auth is enabled but no token provided, fail
    if not provided_token:
        return False
        
    return secrets.compare_digest(provided_token, expected_token)


def extract_federation_token(
    authorization: str | None,
    custom_header: str | None,
) -> str | None:
    """Extract token from Authorization (Bearer) or custom header."""
    if custom_header:
        return custom_header
    
    if authorization:
        scheme, param = get_authorization_scheme_param(authorization)
        if scheme.lower() == "bearer":
            return param
            
    return None