/** @type {import('next').NextConfig} */

const apiBase =
  process.env.NEXT_PUBLIC_API_BASE_EXTERNAL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  `http://localhost:${process.env.BACKEND_PORT || 8001}`;

const nextConfig = {
  publicRuntimeConfig: {
    apiBase,
  },
  // Move dev indicator to bottom-right corner
  devIndicators: {
    position: "bottom-right",
  },

  // Transpile mermaid and related packages for proper ESM handling
  transpilePackages: ["mermaid"],

  // Turbopack configuration (Next.js 16+ uses Turbopack by default for dev)
  turbopack: {
    resolveAlias: {
      // Fix for mermaid's cytoscape dependency - use CJS version
      cytoscape: "cytoscape/dist/cytoscape.cjs.js",
    },
  },

  // Webpack configuration (used for production builds - next build)
  webpack: (config) => {
    const path = require("path");
    config.resolve.alias = {
      ...config.resolve.alias,
      cytoscape: path.resolve(
        __dirname,
        "node_modules/cytoscape/dist/cytoscape.cjs.js",
      ),
    };
    return config;
  },
};

module.exports = nextConfig;
