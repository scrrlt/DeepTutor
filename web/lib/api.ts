// API configuration and utility functions

// Get API base URL from environment variable
// This is automatically set by start_web.py based on config/main.yaml
// The .env.local file is auto-generated on startup with the correct backend port
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE ||
  process.env.NEXT_PUBLIC_API_BASE_EXTERNAL ||
  "http://localhost:8001"; // Fallback for build time

function resolveApiBaseUrl(): string {
  if (typeof window === "undefined") return API_BASE_URL;

  const base = API_BASE_URL;
  const hostname = window.location.hostname;
  const protocol = window.location.protocol; // 'http:' or 'https:'

  if (
    hostname &&
    hostname !== "localhost" &&
    hostname !== "127.0.0.1" &&
    hostname !== "0.0.0.0" &&
    /^http:\/\/localhost(?::\d+)?$/i.test(base)
  ) {
    // Parse API_BASE_URL to extract port
    let port = "";
    try {
      const url = new URL(base);
      port = url.port;
    } catch (e) {
      // If parsing fails, fall back to window.location.port
      port = window.location.port;
    }
    
    // Use the extracted port, or fall back to window.location.port if empty
    const finalPort = port || window.location.port;
    return `${protocol}//${hostname}${finalPort ? `:${finalPort}` : ""}`;
  }

  return base;
}
  }

  return base;
}

/**
 * Construct a full API URL from a path
 * @param path - API path (e.g., '/api/v1/knowledge/list')
 * @returns Full URL (e.g., 'http://localhost:8000/api/v1/knowledge/list')
 */
export function apiUrl(path: string): string {
  // Add a leading slash if missing
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  // Remove trailing slash from base URL if present
  const resolvedBase = resolveApiBaseUrl();
  const base = resolvedBase.endsWith("/") ? resolvedBase.slice(0, -1) : resolvedBase;

  return `${base}${normalizedPath}`;
}

/**
 * Construct a WebSocket URL from a path
 * @param path - WebSocket path (e.g., '/api/v1/solve')
 * @returns WebSocket URL (e.g., 'ws://localhost:{backend_port}/api/v1/solve')
 * Note: backend_port is configured in config/main.yaml
 */
export function wsUrl(path: string): string {
  // Security Hardening: Convert http to ws and https to wss.
  // In production environments (where API_BASE_URL starts with https), this ensures secure websockets.
  const resolvedBase = resolveApiBaseUrl();
  const base = resolvedBase.replace(/^http:/, "ws:").replace(/^https:/, "wss:");

  // Ensure the path has a leading slash
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  // Remove trailing slash from base URL if present
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;

  return `${normalizedBase}${normalizedPath}`;
}
