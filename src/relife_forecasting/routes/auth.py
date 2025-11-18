from fastapi import APIRouter
from relife_forecasting.auth.dependencies import AuthenticatedUserWithRolesDep
from relife_forecasting.models.auth import AuthenticatedUser

router = APIRouter(tags=["auth"])


@router.get("/whoami")
async def whoami_with_roles(
    current_user: AuthenticatedUserWithRolesDep,
) -> AuthenticatedUser:
    """Return authenticated user's information including their Keycloak roles."""

    return current_user
