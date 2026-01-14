import { defineConfig } from "eslint/config";
import globals from "globals";
import nextConfig from "eslint-config-next";

export default defineConfig([
    ...nextConfig,
    {
        languageOptions: {
            globals: {
                ...globals.browser,
            },
            ecmaVersion: 12,
            sourceType: "module",
        },
        rules: {
            semi: ["error", "always"],
            quotes: ["error", "double"],
        },
    }
]);
