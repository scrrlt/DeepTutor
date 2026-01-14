"""User management system for CHIMERA platform."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime, timedelta
import jwt
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    bcrypt = None
    BCRYPT_AVAILABLE = False
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import firebase_admin
    from firebase_admin import auth, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# Mock Firebase for development
if not FIREBASE_AVAILABLE:
    class MockFirebaseAuth:
        def create_user(self, **kwargs): return {"uid": str(uuid.uuid4())}
        def get_user(self, uid): return {"uid": uid, "email": "mock@example.com"}
        def update_user(self, uid, **kwargs): return {"uid": uid}

    class MockFirestore:
        def collection(self, name):
            return MockCollection()

    class MockCollection:
        def doc(self, doc_id):
            return MockDoc()

    class MockDoc:
        def set(self, data): pass
        def get(self): return MockDocSnapshot()
        def update(self, data): pass

    class MockDocSnapshot:
        def exists(self): return True
        def to_dict(self): return {"mock": True}

    auth = MockFirebaseAuth()
    firestore = MockFirestore()

class UserRole(Enum):
    """User role enumeration."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class ResearchFocus(Enum):
    """Research focus areas."""
    ACADEMIC = "academic"
    INDUSTRY = "industry"
    POLICY = "policy"
    GENERAL = "general"

@dataclass
class UserProfile:
    """User profile data structure."""
    uid: str
    email: str
    display_name: Optional[str] = None
    role: UserRole = UserRole.FREE
    research_focus: ResearchFocus = ResearchFocus.ACADEMIC
    preferences: Dict[str, Any] = None
    created_at: datetime = None
    last_login: datetime = None
    api_key: str = None
    usage_stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {
                "email_notifications": True,
                "research_alerts": True,
                "weekly_digest": True,
                "default_word_count": 1500,
                "default_sources": 5,
                "language": "en",
                "timezone": "UTC"
            }
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.api_key is None:
            self.api_key = str(uuid.uuid4()).replace('-', '')
        if self.usage_stats is None:
            self.usage_stats = {
                "total_requests": 0,
                "total_tokens": 0,
                "current_month_requests": 0,
                "current_month_tokens": 0,
                "last_reset": datetime.utcnow().replace(day=1)
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['role'] = self.role.value
        data['research_focus'] = self.research_focus.value
        data['created_at'] = self.created_at.isoformat()
        data['last_login'] = self.last_login.isoformat() if self.last_login else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create from dictionary."""
        data_copy = data.copy()
        data_copy['role'] = UserRole(data.get('role', 'free'))
        data_copy['research_focus'] = ResearchFocus(data.get('research_focus', 'academic'))
        if 'created_at' in data_copy and data_copy['created_at']:
            data_copy['created_at'] = datetime.fromisoformat(data_copy['created_at'])
        if 'last_login' in data_copy and data_copy['last_login']:
            data_copy['last_login'] = datetime.fromisoformat(data_copy['last_login'])
        return cls(**data_copy)

class UserManager:
    """User management system with Firebase integration."""

    def __init__(self):
        self.db = firestore.firestore() if FIREBASE_AVAILABLE else None
        self.jwt_secret = "your-jwt-secret-key"  # In production, use environment variable

    async def create_user(self, email: str, password: str, display_name: Optional[str] = None) -> UserProfile:
        """Create a new user account."""
        try:
            # Create Firebase user
            user_record = auth.create_user(
                email=email,
                password=password,
                display_name=display_name
            )

            # Create user profile
            profile = UserProfile(
                uid=user_record.uid,
                email=email,
                display_name=display_name
            )

            # Store profile in Firestore
            if self.db:
                doc_ref = self.db.collection('users').document(user_record.uid)
                await doc_ref.set(profile.to_dict())

            return profile

        except Exception as e:
            raise Exception(f"Failed to create user: {str(e)}")

    async def get_user(self, uid: str) -> Optional[UserProfile]:
        """Get user profile by UID."""
        try:
            if self.db:
                doc_ref = self.db.collection('users').document(uid)
                doc = await doc_ref.get()
                if doc.exists:
                    return UserProfile.from_dict(doc.to_dict())
            return None
        except Exception as e:
            print(f"Error getting user {uid}: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get user profile by email."""
        try:
            if self.db:
                query = self.db.collection('users').where('email', '==', email).limit(1)
                docs = await query.get()
                for doc in docs:
                    return UserProfile.from_dict(doc.to_dict())
            return None
        except Exception as e:
            print(f"Error getting user by email {email}: {e}")
            return None

    async def update_user(self, uid: str, updates: Dict[str, Any]) -> bool:
        """Update user profile."""
        try:
            if self.db:
                doc_ref = self.db.collection('users').document(uid)
                await doc_ref.update(updates)
                return True
            return False
        except Exception as e:
            print(f"Error updating user {uid}: {e}")
            return False

    async def authenticate_user(self, email: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token."""
        try:
            # In Firebase, we can't verify password directly
            # This would typically be handled by Firebase Auth SDK on client
            # For server-side, we'd use custom tokens or service account auth
            user = await self.get_user_by_email(email)
            if user:
                # Update last login
                await self.update_user(user.uid, {
                    'last_login': datetime.utcnow().isoformat()
                })
                # Generate JWT token
                token = jwt.encode({
                    'uid': user.uid,
                    'email': user.email,
                    'role': user.role.value,
                    'exp': datetime.utcnow() + timedelta(days=7)
                }, self.jwt_secret, algorithm='HS256')
                return token
            return None
        except Exception as e:
            print(f"Authentication error for {email}: {e}")
            return None

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    async def update_usage(self, uid: str, requests: int = 0, tokens: int = 0):
        """Update user usage statistics."""
        try:
            user = await self.get_user(uid)
            if user:
                user.usage_stats['total_requests'] += requests
                user.usage_stats['total_tokens'] += tokens
                user.usage_stats['current_month_requests'] += requests
                user.usage_stats['current_month_tokens'] += tokens

                # Reset monthly counters if needed
                now = datetime.utcnow()
                if now.month != user.usage_stats['last_reset'].month:
                    user.usage_stats['current_month_requests'] = requests
                    user.usage_stats['current_month_tokens'] = tokens
                    user.usage_stats['last_reset'] = now.replace(day=1)

                await self.update_user(uid, {'usage_stats': user.usage_stats})
        except Exception as e:
            print(f"Error updating usage for {uid}: {e}")

    def get_role_limits(self, role: UserRole) -> Dict[str, int]:
        """Get usage limits for a role."""
        limits = {
            UserRole.FREE: {'requests': 100, 'tokens': 10000},
            UserRole.BASIC: {'requests': 1000, 'tokens': 100000},
            UserRole.PROFESSIONAL: {'requests': 10000, 'tokens': 1000000},
            UserRole.ENTERPRISE: {'requests': 100000, 'tokens': 10000000},
            UserRole.ADMIN: {'requests': 1000000, 'tokens': 100000000}
        }
        return limits.get(role, limits[UserRole.FREE])

    async def check_quota(self, uid: str) -> Dict[str, Any]:
        """Check if user is within usage quotas."""
        try:
            user = await self.get_user(uid)
            if not user:
                return {'allowed': False, 'reason': 'User not found'}

            limits = self.get_role_limits(user.role)
            current = user.usage_stats

            if current['current_month_requests'] >= limits['requests']:
                return {
                    'allowed': False,
                    'reason': 'Monthly request limit exceeded',
                    'current': current['current_month_requests'],
                    'limit': limits['requests']
                }

            if current['current_month_tokens'] >= limits['tokens']:
                return {
                    'allowed': False,
                    'reason': 'Monthly token limit exceeded',
                    'current': current['current_month_tokens'],
                    'limit': limits['tokens']
                }

            return {
                'allowed': True,
                'current_requests': current['current_month_requests'],
                'current_tokens': current['current_month_tokens'],
                'limit_requests': limits['requests'],
                'limit_tokens': limits['tokens']
            }

        except Exception as e:
            print(f"Error checking quota for {uid}: {e}")
            return {'allowed': False, 'reason': 'Error checking quota'}

    async def update_preferences(self, uid: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        try:
            user = await self.get_user(uid)
            if user:
                user.preferences.update(preferences)
                await self.update_user(uid, {'preferences': user.preferences})
                return True
            return False
        except Exception as e:
            print(f"Error updating preferences for {uid}: {e}")
            return False

    async def list_users(self, limit: int = 100, offset: int = 0) -> List[UserProfile]:
        """List users (admin function)."""
        try:
            if self.db:
                query = self.db.collection('users').limit(limit).offset(offset)
                docs = await query.get()
                return [UserProfile.from_dict(doc.to_dict()) for doc in docs]
            return []
        except Exception as e:
            print(f"Error listing users: {e}")
            return []


# Global user manager instance
user_manager = UserManager()

# Convenience functions
async def create_user(email: str, password: str, display_name: Optional[str] = None) -> UserProfile:
    """Create a new user."""
    return await user_manager.create_user(email, password, display_name)

async def get_user(uid: str) -> Optional[UserProfile]:
    """Get user by UID."""
    return await user_manager.get_user(uid)

async def authenticate_user(email: str, password: str) -> Optional[str]:
    """Authenticate user and return token."""
    return await user_manager.authenticate_user(email, password)

async def verify_user_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token."""
    return await user_manager.verify_token(token)

async def update_user_usage(uid: str, requests: int = 0, tokens: int = 0):
    """Update user usage."""
    await user_manager.update_usage(uid, requests, tokens)

async def check_user_quota(uid: str) -> Dict[str, Any]:
    """Check user quota."""
    return await user_manager.check_quota(uid)


if __name__ == "__main__":
    # Test user management
    import asyncio

    async def test():
        print("Testing user management...")

        # Test creating a user
        try:
            user = await create_user("test@example.com", "password123", "Test User")
            print(f"Created user: {user.uid}")

            # Test getting user
            retrieved = await get_user(user.uid)
            print(f"Retrieved user: {retrieved.email if retrieved else 'None'}")

            # Test authentication
            token = await authenticate_user("test@example.com", "password123")
            print(f"Auth token: {token[:20] if token else 'None'}...")

            # Test quota check
            quota = await check_user_quota(user.uid)
            print(f"Quota status: {quota}")

        except Exception as e:
            print(f"Test error: {e}")

    asyncio.run(test())