"""
Authentication and authorization
================================
JWT auth with per-user data scoping and a role for knowledge-base admins.

The previous API had no authentication of any kind, and `session_id` was
supplied by the client and trusted. Combined with `GET /api/sessions`, which
returned the twenty most recent session IDs to anyone, that made every
student's chat history and uploaded documents readable — and deletable — by
any visitor:

    GET  /api/sessions                     -> harvest IDs
    GET  /api/sessions/<id>                -> read that person's transcript
    DELETE /api/sessions/<id>              -> destroy it

Identity is the only real fix. A session now belongs to a user id taken from
a signed token, never from the request body, and every read and write is
scoped to the caller.

Design notes:

  * Access tokens are short-lived and stateless; refresh tokens are long-lived
    and stored server-side so logout can actually revoke them. A stateless
    refresh token cannot be revoked, which makes "sign out everywhere"
    impossible.
  * Passwords use bcrypt with a per-password salt.
  * `verify_password` runs a dummy hash when the user does not exist, so
    response timing does not reveal which accounts are registered.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

import bcrypt
import jwt
from sqlalchemy import Engine, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from edupilot.db.tables import refresh_tokens, users

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TOKEN_TTL = timedelta(minutes=30)
REFRESH_TOKEN_TTL = timedelta(days=14)

#: bcrypt truncates at 72 bytes; reject longer rather than silently ignoring
#: the tail, which would make two different long passwords equivalent.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 10

#: Cost factor. 12 is ~250ms on current hardware — slow enough to matter for
#: offline cracking, fast enough for interactive login.
BCRYPT_ROUNDS = 12

#: Precomputed hash used to equalize timing on unknown-user login attempts.
_DUMMY_HASH = bcrypt.hashpw(b"timing-equalization-placeholder", bcrypt.gensalt(BCRYPT_ROUNDS))


class Role(str, Enum):
    """
    Authorization roles.

    `str, Enum` rather than `StrEnum` so this runs on Python 3.10, where
    `enum.StrEnum` does not exist. The mixin gives the same behaviour that
    matters here: a Role serializes as its value in JSON and compares equal
    to the plain string.
    """

    STUDENT = "student"
    """Can chat and manage their own study sessions and uploads."""
    ADMIN = "admin"
    """Additionally can modify the shared course knowledge base."""

    def __str__(self) -> str:
        return self.value


class AuthError(Exception):
    """Authentication or authorization failure."""

    def __init__(self, message: str, *, status: int = 401) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class User:
    """An authenticated principal."""

    user_id: str
    email: str
    role: Role
    display_name: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    def namespace(self) -> str:
        """Per-user vector namespace prefix, keeping study uploads isolated."""
        return f"u_{self.user_id.replace('-', '')}"


# ---------------------------------------------------------------------------
# Secret
# ---------------------------------------------------------------------------


def get_secret_key() -> str:
    """
    Load the JWT signing secret.

    Refuses to start without one in production. An auto-generated fallback in
    development is a convenience; the same behaviour in production would mean
    every restart silently invalidated all tokens, and a hardcoded default
    would mean anyone could mint an admin token.
    """
    secret = os.getenv("JWT_SECRET_KEY", "")
    if secret:
        if len(secret) < 32:
            raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters")
        return secret

    if os.getenv("EDUPILOT_ENV", "development").lower() == "production":
        raise RuntimeError(
            "JWT_SECRET_KEY is required in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )

    logger.warning(
        "JWT_SECRET_KEY not set — using an ephemeral development key. "
        "All tokens become invalid when this process restarts."
    )
    return secrets.token_urlsafe(48)


_SECRET: str | None = None
_secret_lock = threading.Lock()


def _secret() -> str:
    global _SECRET
    if _SECRET is None:
        with _secret_lock:
            if _SECRET is None:
                _SECRET = get_secret_key()
    return _SECRET


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash a password with bcrypt. Raises ValueError if it fails policy."""
    encoded = password.encode("utf-8")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes "
            "(bcrypt silently truncates beyond this)."
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt(BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str | None) -> bool:
    """
    Check a password against its hash in constant-ish time.

    When `password_hash` is None (no such user) a dummy comparison still runs,
    so a missing account and a wrong password take the same time. Skipping it
    turns login into a user-enumeration oracle.
    """
    encoded = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    if password_hash is None:
        bcrypt.checkpw(encoded, _DUMMY_HASH)
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        logger.warning("malformed password hash encountered")
        return False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def create_access_token(user: User, *, ttl: timedelta = ACCESS_TOKEN_TTL) -> str:
    """Mint a short-lived stateless access token."""
    now = datetime.now(UTC)
    payload = {
        "sub": user.user_id,
        "email": user.email,
        "role": str(user.role),
        "name": user.display_name,
        "type": "access",
        "iat": now,
        "exp": now + ttl,
        "jti": secrets.token_urlsafe(8),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def create_refresh_token() -> tuple[str, str, datetime]:
    """
    Mint a refresh token.

    Returns (token, token_hash, expires_at). Only the hash is stored, so a
    database leak does not hand out usable refresh tokens.
    """
    import hashlib

    token = secrets.token_urlsafe(48)
    digest = hashlib.sha256(token.encode()).hexdigest()
    return token, digest, datetime.now(UTC) + REFRESH_TOKEN_TTL


def decode_access_token(token: str) -> User:
    """
    Verify an access token and return its principal.

    Raises:
        AuthError: expired, malformed, wrong type, or bad signature.
    """
    try:
        payload = jwt.decode(
            token,
            _secret(),
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Session expired. Please sign in again.") from None
    except jwt.InvalidTokenError as exc:
        logger.info("rejected token: %s", exc)
        raise AuthError("Invalid authentication token.") from None

    # An access endpoint must not accept a refresh token.
    if payload.get("type") != "access":
        raise AuthError("Invalid token type.")

    try:
        role = Role(payload.get("role", Role.STUDENT))
    except ValueError:
        role = Role.STUDENT

    return User(
        user_id=str(payload["sub"]),
        email=str(payload.get("email", "")),
        role=role,
        display_name=str(payload.get("name", "")),
    )


# ---------------------------------------------------------------------------
# User store
# ---------------------------------------------------------------------------


class UserStore:
    """
    User and refresh-token storage, in Postgres.

    Args:
        engine: The SQLAlchemy engine. Defaults to the app's shared one.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        from edupilot.db import get_engine

        self.engine = engine or get_engine()

    # -- users ------------------------------------------------------------

    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()

    @staticmethod
    def _user(row) -> User:
        return User(
            user_id=row.user_id,
            email=row.email,
            role=Role(row.role),
            display_name=row.display_name,
        )

    def create_user(
        self, email: str, password: str, *, role: Role = Role.STUDENT, display_name: str = ""
    ) -> User:
        """
        Register a user.

        Raises:
            AuthError: the email is already registered.
            ValueError: the password fails policy.
        """
        email = self.normalize_email(email)
        if not email or "@" not in email:
            raise ValueError("A valid email address is required.")

        password_hash = hash_password(password)
        user_id = str(uuid.uuid4())
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(users).values(
                        user_id=user_id,
                        email=email,
                        password_hash=password_hash,
                        role=str(role),
                        display_name=display_name or email.split("@")[0],
                    )
                )
        except IntegrityError:
            raise AuthError("An account with that email already exists.", status=409) from None

        logger.info("registered user %s (%s)", user_id, role)
        return User(user_id=user_id, email=email, role=role, display_name=display_name)

    def authenticate(self, email: str, password: str) -> User:
        """
        Verify credentials.

        Raises:
            AuthError: with a message that does not reveal whether the account
                exists — distinguishing "no such user" from "wrong password"
                hands an attacker a list of valid accounts.
        """
        with self.engine.connect() as conn:
            row = conn.execute(
                select(users).where(users.c.email == self.normalize_email(email))
            ).first()

        stored_hash = row.password_hash if row else None
        if not verify_password(password, stored_hash) or not row:
            raise AuthError("Incorrect email or password.")
        if not row.is_active:
            raise AuthError("This account has been deactivated.", status=403)

        with self.engine.begin() as conn:
            conn.execute(
                update(users).where(users.c.user_id == row.user_id).values(last_login_at=func.now())
            )
        return self._user(row)

    def get_user(self, user_id: str) -> User | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(users).where(users.c.user_id == user_id, users.c.is_active)
            ).first()
        return self._user(row) if row else None

    def find_user_id(self, email: str) -> str | None:
        """The user id registered to `email`, or None."""
        with self.engine.connect() as conn:
            return conn.execute(
                select(users.c.user_id).where(users.c.email == self.normalize_email(email))
            ).scalar()

    def count_users(self) -> int:
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(users)).scalar_one()

    # -- refresh tokens ---------------------------------------------------

    def store_refresh_token(self, user_id: str, token_hash: str, expires_at: datetime) -> None:
        statement = pg_insert(refresh_tokens).values(
            token_hash=token_hash, user_id=user_id, expires_at=expires_at
        )
        statement = statement.on_conflict_do_update(
            index_elements=["token_hash"],
            set_={"user_id": user_id, "expires_at": expires_at, "revoked": False},
        )
        with self.engine.begin() as conn:
            conn.execute(statement)

    def consume_refresh_token(self, token: str) -> User:
        """
        Exchange a refresh token for its user, rotating it out.

        Single-use: the token is revoked on redemption, so a stolen token is
        usable at most once and reuse is detectable. Checking and revoking is
        one conditional UPDATE, so two requests racing with the same token —
        on different API instances, say — cannot both redeem it.
        """
        import hashlib

        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.engine.begin() as conn:
            row = conn.execute(
                update(refresh_tokens)
                .where(refresh_tokens.c.token_hash == digest, refresh_tokens.c.revoked.is_(False))
                .values(revoked=True)
                .returning(refresh_tokens.c.user_id, refresh_tokens.c.expires_at)
            ).first()

        if not row:
            raise AuthError("Invalid or already-used refresh token.")
        if row.expires_at < datetime.now(UTC):
            raise AuthError("Refresh token expired. Please sign in again.")

        user = self.get_user(row.user_id)
        if not user:
            raise AuthError("Account no longer exists.")
        return user

    def revoke_all_refresh_tokens(self, user_id: str) -> int:
        """Sign out everywhere. Returns the number of tokens revoked."""
        with self.engine.begin() as conn:
            result = conn.execute(
                update(refresh_tokens)
                .where(refresh_tokens.c.user_id == user_id, refresh_tokens.c.revoked.is_(False))
                .values(revoked=True)
            )
            return result.rowcount

    def purge_expired_tokens(self) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(refresh_tokens).where(refresh_tokens.c.expires_at < func.now())
            )
            return result.rowcount
