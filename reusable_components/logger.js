/**
 * Comprehensive logging system for CHIMERA Platform
 * Lightweight implementation without external dependencies
 */

const fs = require("fs");
const path = require("path");
const os = require("os");

// Ensure logs directory exists
const logsDir = path.join(process.cwd(), "logs");
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// Log levels
const LOG_LEVELS = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
  trace: 4,
};

const LOG_LEVEL_NAMES = Object.keys(LOG_LEVELS);

// Log colors for console output (ANSI escape codes)
const LOG_COLORS = {
  error: "\x1b[31m", // Red
  warn: "\x1b[33m", // Yellow
  info: "\x1b[36m", // Cyan
  debug: "\x1b[35m", // Magenta
  trace: "\x1b[37m", // White
  reset: "\x1b[0m",
};

// Current log level (can be set via environment)
let currentLogLevel = LOG_LEVELS[process.env.LOG_LEVEL || "info"];

// File streams for logging
let errorStream, combinedStream, performanceStream;

function initializeFileStreams() {
  if (!errorStream) {
    errorStream = fs.createWriteStream(path.join(logsDir, "error.log"), {
      flags: "a",
    });
  }
  if (!combinedStream) {
    combinedStream = fs.createWriteStream(path.join(logsDir, "combined.log"), {
      flags: "a",
    });
  }
  if (!performanceStream) {
    performanceStream = fs.createWriteStream(
      path.join(logsDir, "performance.log"),
      { flags: "a" }
    );
  }
}

// Initialize streams
initializeFileStreams();

// Format log entry as JSON
function formatLogEntry(level, message, metadata = {}) {
  const logEntry = {
    timestamp: new Date().toISOString(),
    level: level.toUpperCase(),
    message,
    service: "chimera-platform",
    hostname: os.hostname(),
    pid: process.pid,
    version: process.env.npm_package_version || "1.0.0",
    environment: process.env.NODE_ENV || "development",
    ...metadata,
  };

  // Add request context if available
  if (metadata.req) {
    logEntry.request = {
      method: metadata.req.method,
      url: metadata.req.url,
      userAgent: metadata.req.get("User-Agent"),
      ip: metadata.req.ip || metadata.req.connection?.remoteAddress,
    };
    delete metadata.req;
  }

  // Add user context if available
  if (metadata.userId) {
    logEntry.user = {
      id: metadata.userId,
      apiKey: metadata.apiKey
        ? metadata.apiKey.substring(0, 8) + "..."
        : undefined,
    };
  }

  // Add performance metrics if available
  if (metadata.duration) {
    logEntry.performance = {
      duration: metadata.duration,
      endpoint: metadata.endpoint,
    };
  }

  return logEntry;
}

// Write to file
function writeToFile(stream, entry) {
  try {
    stream.write(JSON.stringify(entry) + "\n");
  } catch (error) {
    console.error("Failed to write to log file:", error);
  }
}

// Write to BigQuery (if available)
async function writeToBigQuery(entry) {
  try {
    // Try to load BigQuery transport
    const BigQueryTransport = require("./transports/bigquery-transport");
    if (BigQueryTransport) {
      const transport = new BigQueryTransport();
      await transport.log(entry, () => {});
    }
  } catch (error) {
    // BigQuery not available, skip
  }
}

// Core logging function
async function log(level, message, metadata = {}) {
  if (LOG_LEVELS[level] > currentLogLevel) {
    return;
  }

  const entry = formatLogEntry(level, message, metadata);

  // Write to appropriate files
  if (level === "error") {
    writeToFile(errorStream, entry);
  }
  writeToFile(combinedStream, entry);

  if (metadata.performance || metadata.duration) {
    writeToFile(performanceStream, entry);
  }

  // Write to BigQuery if configured
  if (process.env.BQ_DATASET_ID) {
    writeToBigQuery(entry);
  }

  // Console output for development
  if (process.env.NODE_ENV !== "production") {
    const color = LOG_COLORS[level] || LOG_COLORS.reset;
    const timestamp = entry.timestamp.slice(11, 19); // HH:MM:SS
    const levelStr = level.toUpperCase().padEnd(5);
    const metaStr =
      Object.keys(metadata).length > 0 ? ` ${JSON.stringify(metadata)}` : "";
    console.log(
      `${color}${timestamp} ${levelStr} ${message}${metaStr}${LOG_COLORS.reset}`
    );
  }
}

// Logger class for contextual logging
class ChimeraLogger {
  constructor(context = {}) {
    this.context = context;
  }

  async error(message, meta = {}) {
    await log("error", message, { ...this.context, ...meta });
  }

  async warn(message, meta = {}) {
    await log("warn", message, { ...this.context, ...meta });
  }

  async info(message, meta = {}) {
    await log("info", message, { ...this.context, ...meta });
  }

  async debug(message, meta = {}) {
    await log("debug", message, { ...this.context, ...meta });
  }

  async trace(message, meta = {}) {
    await log("trace", message, { ...this.context, ...meta });
  }

  // Specialized logging methods
  async logRequest(req, res, duration) {
    const meta = {
      req,
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration,
      userAgent: req.get("User-Agent"),
      ip: req.ip || req.connection?.remoteAddress,
      requestId: req.requestId,
    };

    if (res.statusCode >= 400) {
      await this.error(`Request failed: ${req.method} ${req.url}`, meta);
    } else {
      await this.info(`Request completed: ${req.method} ${req.url}`, meta);
    }
  }

  async logDatabase(operation, duration, success = true, meta = {}) {
    const logData = {
      operation,
      duration,
      success,
      ...meta,
    };

    if (success) {
      await this.debug(`Database operation: ${operation}`, logData);
    } else {
      await this.error(`Database operation failed: ${operation}`, logData);
    }
  }

  async logApiCall(endpoint, method, statusCode, duration, meta = {}) {
    const logData = {
      endpoint,
      method,
      statusCode,
      duration,
      ...meta,
    };

    if (statusCode >= 400) {
      await this.warn(`API call failed: ${method} ${endpoint}`, logData);
    } else {
      await this.debug(`API call: ${method} ${endpoint}`, logData);
    }
  }

  async logPerformance(operation, duration, meta = {}) {
    await this.info(`Performance: ${operation}`, {
      operation,
      duration,
      performance: true,
      ...meta,
    });
  }

  async logSecurity(event, severity = "info", meta = {}) {
    const logData = {
      event,
      severity,
      ...meta,
    };

    if (severity === "high" || severity === "critical") {
      await this.error(`Security event: ${event}`, logData);
    } else if (severity === "medium") {
      await this.warn(`Security event: ${event}`, logData);
    } else {
      await this.info(`Security event: ${event}`, logData);
    }
  }

  async logBusiness(event, meta = {}) {
    await this.info(`Business event: ${event}`, {
      eventType: "business",
      ...meta,
    });
  }

  // Create child logger with additional context
  child(additionalContext = {}) {
    return new ChimeraLogger({ ...this.context, ...additionalContext });
  }
}

// Request logging middleware
function requestLogger(options = {}) {
  const logger = new ChimeraLogger({ component: "http" });

  return (req, res, next) => {
    const start = Date.now();
    const requestId = req.headers["x-request-id"] || generateRequestId();

    // Add request ID to request object
    req.requestId = requestId;

    // Log request start
    logger.debug(`Request started: ${req.method} ${req.url}`, {
      method: req.method,
      url: req.url,
      userAgent: req.get("User-Agent"),
      ip: req.ip,
      requestId,
    });

    // Log response
    res.on("finish", () => {
      const duration = Date.now() - start;
      logger.logRequest(req, res, duration);
    });

    next();
  };
}

// Performance logging middleware
function performanceLogger(threshold = 1000) {
  const logger = new ChimeraLogger({ component: "performance" });

  return (req, res, next) => {
    const start = Date.now();

    const originalEnd = res.end;
    res.end = function (...args) {
      const duration = Date.now() - start;

      if (duration > threshold) {
        logger.warn(`Slow request: ${req.method} ${req.url}`, {
          method: req.method,
          url: req.url,
          duration,
          threshold,
        });
      }

      originalEnd.apply(this, args);
    };

    next();
  };
}

// Error logging middleware
function errorLogger() {
  const logger = new ChimeraLogger({ component: "error" });

  return (err, req, res, next) => {
    logger.error("Unhandled error", {
      error: err.message,
      stack: err.stack,
      method: req.method,
      url: req.url,
      headers: req.headers,
      userId: req.userId,
      apiKey: req.apiKey,
      requestId: req.requestId,
    });

    next(err);
  };
}

// Specialized logging functions
async function logPerformance(operation, duration, metadata = {}) {
  const logger = new ChimeraLogger({ component: "performance" });
  await logger.logPerformance(operation, duration, metadata);
}

async function logError(error, context = {}) {
  const logger = new ChimeraLogger({ component: "error" });
  await logger.error(error.message, {
    stack: error.stack,
    ...context,
    error: true,
  });
}

async function logUserAction(userId, action, metadata = {}) {
  const logger = new ChimeraLogger({ component: "user" });
  await logger.info(`User action: ${action}`, {
    userId,
    action,
    userAction: true,
    ...metadata,
  });
}

async function logApiCall(
  endpoint,
  method,
  statusCode,
  duration,
  metadata = {}
) {
  const logger = new ChimeraLogger({ component: "api" });
  await logger.logApiCall(endpoint, method, statusCode, duration, metadata);
}

// Generate unique request ID
function generateRequestId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Flush logs (useful for graceful shutdown)
async function flushLogs() {
  return new Promise((resolve) => {
    // Close file streams
    if (errorStream) errorStream.end();
    if (combinedStream) combinedStream.end();
    if (performanceStream) performanceStream.end();

    // Wait a bit for writes to complete
    setTimeout(resolve, 100);
  });
}

// Get logger stats
function getLoggerStats() {
  return {
    level: LOG_LEVEL_NAMES[currentLogLevel],
    logLevel: currentLogLevel,
    streams: {
      error: !!errorStream,
      combined: !!combinedStream,
      performance: !!performanceStream,
    },
    bigQueryEnabled: !!process.env.BQ_DATASET_ID,
  };
}

// Set log level
function setLogLevel(level) {
  if (typeof level === "string" && LOG_LEVELS[level] !== undefined) {
    currentLogLevel = LOG_LEVELS[level];
  } else if (typeof level === "number" && level >= 0 && level <= 4) {
    currentLogLevel = level;
  }
}

// Export logger instance and utilities
const defaultLogger = new ChimeraLogger();

module.exports = {
  ChimeraLogger,
  logger: defaultLogger,
  requestLogger,
  performanceLogger,
  errorLogger,
  logPerformance,
  logError,
  logUserAction,
  logApiCall,
  generateRequestId,
  flushLogs,
  getLoggerStats,
  setLogLevel,
  LOG_LEVELS,
};
