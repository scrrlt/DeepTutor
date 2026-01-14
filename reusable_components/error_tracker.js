/**
 * Error tracking for CHIMERA Platform
 * Simple implementation - can be extended with Sentry, Rollbar, etc.
 */

class ErrorTracker {
  constructor() {
    this.errors = [];
    this.maxErrors = 1000;
  }

  track_error(error, context = {}, userId = null, tags = {}) {
    const errorEntry = {
      error: error.message || error,
      stack: error.stack,
      context,
      userId,
      tags,
      timestamp: new Date().toISOString(),
      userAgent: context.userAgent || "unknown",
      url: context.url || "unknown",
    };

    this.errors.push(errorEntry);

    // Keep only recent errors
    if (this.errors.length > this.maxErrors) {
      this.errors.shift();
    }

    // Log to console for now
    console.error("Tracked error:", {
      message: error.message,
      userId,
      endpoint: tags.endpoint,
      timestamp: errorEntry.timestamp,
    });

    // In production, send to error tracking service
    // this.sendToService(errorEntry);
  }

  getRecentErrors(limit = 50) {
    return this.errors.slice(-limit);
  }

  getErrorStats() {
    const stats = {
      total: this.errors.length,
      byEndpoint: {},
      byUser: {},
      recent: [],
    };

    this.errors.forEach((error) => {
      const endpoint = error.tags.endpoint || "unknown";
      const userId = error.userId || "anonymous";

      stats.byEndpoint[endpoint] = (stats.byEndpoint[endpoint] || 0) + 1;
      stats.byUser[userId] = (stats.byUser[userId] || 0) + 1;
    });

    stats.recent = this.errors.slice(-10);
    return stats;
  }

  // Placeholder for external service integration
  async sendToService(errorEntry) {
    // Implement integration with Sentry, Rollbar, etc.
    // For now, just log
    console.log("Would send to error tracking service:", errorEntry);
  }
}

// Global error tracker instance
const errorTracker = new ErrorTracker();

module.exports = {
  ErrorTracker,
  errorTracker,
};
