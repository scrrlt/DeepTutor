declare namespace NodeJS {
  /** Minimal ProcessEnv shape to avoid depending on @types/node */
  interface ProcessEnv {
    [key: string]: string | undefined;
    NODE_ENV?: "development" | "production" | "test";
    NEXT_PUBLIC_API_BASE?: string;
    WEB_BASE_URL?: string;
    PW_SERIAL?: string;
    CI?: string;
  }

  interface Process {
    env: ProcessEnv;
    cwd(): string;
    exit(code?: number): never;
  }

  type Timeout = number;
}

declare const process: NodeJS.Process;
