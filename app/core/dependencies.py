"""
==============================================================================
FastAPI Dependencies — RBAC, Authentication, and Dependency Injection
==============================================================================

ARCHITECTURAL DECISION (Role-Based Access Control):
    Banking systems require strict access control. We implement RBAC using
    FastAPI's dependency injection system, which provides:
    
    1. **Declarative security** — Endpoints declare their required role in
       the function signature, not in middleware. This makes access control
       visible, auditable, and testable.
       
    2. **Layered enforcement** — Dependencies compose:
       get_current_user() → require_role("admin") → endpoint handler
       Each layer can reject the request independently.
       
    3. **Separation of concerns** — Auth logic lives here, business logic
       lives in route handlers. A security auditor can review this one file
       to understand the entire access control model.

ROLE HIERARCHY:
    ┌──────────────┐
    │   system     │ ← Internal services only (KB token generation)
    ├──────────────┤
    │   evaluator  │ ← Assessment evaluators (debug endpoints)
    ├──────────────┤
    │   admin      │ ← Bank administrators (logs, chunk inspection)
    ├──────────────┤
    │   user       │ ← Standard bank employees (query only)
    └──────────────┘

    Higher roles inherit lower role permissions (system > evaluator > admin > user).

IMPORTANT NOTE:
    For this assessment, we use header-based role simulation (X-User-Role).
    In production, this would be replaced by:
    - OAuth2 / OpenID Connect tokens from an identity provider
    - Service mesh mTLS for internal service auth
    - API Gateway enforced policies (Kong, AWS API Gateway, etc.)
"""

from enum import Enum
from typing import Optional
from fastapi import Header, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from loguru import logger

from app.core.security import validate_kb_token


# =============================================================================
# Role Definitions
# =============================================================================

class UserRole(str, Enum):
    """
    Enumerated roles with hierarchy. Using str mixin for JSON serialization.
    
    Role assignments in production would come from:
    - LDAP/Active Directory groups
    - OAuth2 scope claims
    - Database role mappings
    """
    USER = "user"
    ADMIN = "admin"
    EVALUATOR = "evaluator"
    SYSTEM = "system"


# Role hierarchy — maps each role to its effective permissions
# A "system" role can do everything a "user" can, plus more.
ROLE_HIERARCHY: dict[UserRole, int] = {
    UserRole.USER: 1,
    UserRole.ADMIN: 2,
    UserRole.EVALUATOR: 3,
    UserRole.SYSTEM: 4,
}


class CurrentUser(BaseModel):
    """
    Represents the authenticated user context.
    Injected into every request handler via FastAPI's Depends().
    """
    username: str
    role: UserRole
    is_authenticated: bool = True


# =============================================================================
# Authentication Dependencies
# =============================================================================

# HTTP Bearer scheme for KB token validation
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    x_user_role: str = Header(
        default="user",
        description=(
            "Simulated user role for RBAC. In production, this would be "
            "extracted from a validated OAuth2/JWT token. "
            "Accepted values: user, admin, evaluator, system"
        ),
    ),
    x_username: str = Header(
        default="anonymous",
        description="Simulated username for audit logging",
    ),
) -> CurrentUser:
    """
    Extract and validate the current user context from request headers.
    
    WHY HEADER-BASED SIMULATION?
    For this assessment, we simulate RBAC via headers to make the system
    testable without an identity provider. The architecture is designed so
    that replacing this function with real OAuth2 validation requires
    changing ONLY this dependency — no endpoint code changes needed.
    
    In production, this dependency would:
    1. Extract the Bearer token from the Authorization header
    2. Validate it against the identity provider's JWKS endpoint
    3. Extract roles from token claims
    4. Return a CurrentUser with real identity data
    """
    # Validate the role string
    try:
        role = UserRole(x_user_role.lower())
    except ValueError:
        logger.warning(f"Invalid role attempted: {x_user_role} by {x_username}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_role",
                "message": f"Role '{x_user_role}' is not recognized. "
                           f"Valid roles: {[r.value for r in UserRole]}",
            },
        )

    user = CurrentUser(username=x_username, role=role)
    logger.debug(f"Authenticated user: {user.username} with role: {user.role.value}")
    return user


# =============================================================================
# Role Enforcement Dependencies (Composable)
# =============================================================================

def require_role(minimum_role: UserRole):
    """
    Factory function that creates a FastAPI dependency requiring a minimum role.
    
    Uses the role hierarchy to determine if the user's role is sufficient.
    This pattern allows endpoint-level access control:
    
        @router.get("/admin-only", dependencies=[Depends(require_role(UserRole.ADMIN))])
        async def admin_endpoint():
            ...
    
    Args:
        minimum_role: The minimum role required to access the endpoint
    
    Returns:
        A FastAPI dependency function that raises 403 if insufficient
    """
    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        user_level = ROLE_HIERARCHY.get(current_user.role, 0)
        required_level = ROLE_HIERARCHY.get(minimum_role, 999)

        if user_level < required_level:
            logger.warning(
                f"ACCESS DENIED | user={current_user.username} | "
                f"role={current_user.role.value} | required={minimum_role.value} | "
                f"endpoint requires elevated privileges"
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "insufficient_permissions",
                    "message": (
                        f"This endpoint requires '{minimum_role.value}' role or higher. "
                        f"Your current role '{current_user.role.value}' is insufficient."
                    ),
                    "required_role": minimum_role.value,
                    "your_role": current_user.role.value,
                },
            )

        logger.info(
            f"ACCESS GRANTED | user={current_user.username} | "
            f"role={current_user.role.value} | required={minimum_role.value}"
        )
        return current_user

    return role_checker


# =============================================================================
# KB Token Validation Dependency
# =============================================================================

async def validate_kb_access(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> dict:
    """
    Validate KB access token from the Authorization header.
    
    This dependency is used exclusively on the `/kb/fetch` endpoint
    to ensure that KB data can only be accessed with a valid, non-expired
    token obtained from `/kb/token`.
    
    Flow:
    1. Extract Bearer token from Authorization header
    2. Decode and validate JWT (signature + expiry + purpose claim)
    3. Return decoded payload for downstream use
    
    WHY A SEPARATE DEPENDENCY (not the same as user auth)?
    KB access tokens serve a different purpose than user authentication:
    - User auth: "Who are you?" → identity verification
    - KB token: "Are you allowed to access this resource right now?" → 
      capability-based access control with short-lived authorization
    
    This separation follows the Principle of Least Privilege — even an
    authenticated admin cannot access KB data without a fresh token.
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_kb_token",
                "message": (
                    "KB access requires a valid Bearer token. "
                    "Obtain one from POST /kb/token first."
                ),
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = validate_kb_token(credentials.credentials)

    if not result.is_valid:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_kb_token",
                "message": result.error,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return result.payload


# =============================================================================
# Pre-built Role Dependencies (Convenience)
# =============================================================================
# These are ready-to-use dependencies for common role requirements.
# Usage: Depends(require_admin)

require_user = require_role(UserRole.USER)
require_admin = require_role(UserRole.ADMIN)
require_evaluator = require_role(UserRole.EVALUATOR)
require_system = require_role(UserRole.SYSTEM)
