/**
 * OAuth Configuration for CHIMERA Platform
 * Supports Google, GitHub, and Microsoft OAuth providers
 */

const oauthConfig = {
  google: {
    clientId: process.env.GOOGLE_OAUTH_CLIENT_ID,
    clientSecret: process.env.GOOGLE_OAUTH_CLIENT_SECRET,
    redirectUri:
      process.env.GOOGLE_OAUTH_REDIRECT_URI ||
      `${
        process.env.BASE_URL || "http://localhost:3001"
      }/api/auth/oauth/google/callback`,
    authorizationUrl: "https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl: "https://oauth2.googleapis.com/token",
    userInfoUrl: "https://www.googleapis.com/oauth2/v2/userinfo",
    scope: "openid email profile",
    provider: "google",
  },
  github: {
    clientId: process.env.GITHUB_OAUTH_CLIENT_ID,
    clientSecret: process.env.GITHUB_OAUTH_CLIENT_SECRET,
    redirectUri:
      process.env.GITHUB_OAUTH_REDIRECT_URI ||
      `${
        process.env.BASE_URL || "http://localhost:3001"
      }/api/auth/oauth/github/callback`,
    authorizationUrl: "https://github.com/login/oauth/authorize",
    tokenUrl: "https://github.com/login/oauth/access_token",
    userInfoUrl: "https://api.github.com/user",
    scope: "user:email",
    provider: "github",
  },
  microsoft: {
    clientId: process.env.MICROSOFT_OAUTH_CLIENT_ID,
    clientSecret: process.env.MICROSOFT_OAUTH_CLIENT_SECRET,
    redirectUri:
      process.env.MICROSOFT_OAUTH_REDIRECT_URI ||
      `${
        process.env.BASE_URL || "http://localhost:3001"
      }/api/auth/oauth/microsoft/callback`,
    authorizationUrl:
      "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
    tokenUrl: "https://login.microsoftonline.com/common/oauth2/v2.0/token",
    userInfoUrl: "https://graph.microsoft.com/v1.0/me",
    scope: "openid email profile",
    provider: "microsoft",
  },
};

// Validate OAuth configuration
function validateOAuthConfig() {
  const providers = Object.keys(oauthConfig);
  const missing = [];

  providers.forEach((provider) => {
    const config = oauthConfig[provider];
    if (!config.clientId) {
      missing.push(`${provider.toUpperCase()}_OAUTH_CLIENT_ID`);
    }
    if (!config.clientSecret) {
      missing.push(`${provider.toUpperCase()}_OAUTH_CLIENT_SECRET`);
    }
  });

  if (missing.length > 0) {
    console.warn(
      "OAuth configuration incomplete. Missing environment variables:",
      missing.join(", ")
    );
    console.warn("OAuth providers will not be available until configured.");
  }

  return missing.length === 0;
}

// Generate OAuth authorization URL
function getAuthorizationUrl(provider, state = null) {
  const config = oauthConfig[provider];
  if (!config) {
    throw new Error(`Unsupported OAuth provider: ${provider}`);
  }

  const params = new URLSearchParams({
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    scope: config.scope,
    response_type: "code",
    ...(state && { state }),
  });

  return `${config.authorizationUrl}?${params.toString()}`;
}

// Exchange authorization code for access token
async function exchangeCodeForToken(provider, code) {
  const config = oauthConfig[provider];
  if (!config) {
    throw new Error(`Unsupported OAuth provider: ${provider}`);
  }

  const response = await fetch(config.tokenUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Accept: "application/json",
      ...(provider === "github" && { Accept: "application/json" }),
    },
    body: new URLSearchParams({
      client_id: config.clientId,
      client_secret: config.clientSecret,
      code,
      redirect_uri: config.redirectUri,
      grant_type: "authorization_code",
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OAuth token exchange failed: ${error}`);
  }

  return response.json();
}

// Get user info from OAuth provider
async function getUserInfo(provider, accessToken) {
  const config = oauthConfig[provider];
  if (!config) {
    throw new Error(`Unsupported OAuth provider: ${provider}`);
  }

  const headers = {
    Authorization: `Bearer ${accessToken}`,
    Accept: "application/json",
  };

  // GitHub requires User-Agent header
  if (provider === "github") {
    headers["User-Agent"] = "CHIMERA-Platform";
  }

  const response = await fetch(config.userInfoUrl, { headers });

  if (!response.ok) {
    throw new Error(`Failed to get user info from ${provider}`);
  }

  const userInfo = await response.json();

  // Normalize user info across providers
  return normalizeUserInfo(provider, userInfo);
}

// Normalize user info to consistent format
function normalizeUserInfo(provider, userInfo) {
  switch (provider) {
    case "google":
      return {
        id: userInfo.id,
        email: userInfo.email,
        name: userInfo.name,
        picture: userInfo.picture,
        provider: "google",
        verified: userInfo.verified_email,
      };

    case "github":
      return {
        id: userInfo.id.toString(),
        email: userInfo.email,
        name: userInfo.name || userInfo.login,
        picture: userInfo.avatar_url,
        provider: "github",
        verified: true, // GitHub emails are verified
      };

    case "microsoft":
      return {
        id: userInfo.id,
        email: userInfo.mail || userInfo.userPrincipalName,
        name: userInfo.displayName,
        picture: null, // Microsoft Graph doesn't provide avatar URL in basic scope
        provider: "microsoft",
        verified: true, // Microsoft accounts are verified
      };

    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

module.exports = {
  oauthConfig,
  validateOAuthConfig,
  getAuthorizationUrl,
  exchangeCodeForToken,
  getUserInfo,
  normalizeUserInfo,
};
