import React from 'react';

interface KBErrorDisplayProps {
  error: string | null;
}

export function KBErrorDisplay({ error }: KBErrorDisplayProps) {
  if (!error) return null;

  return (
    <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded border border-red-200 dark:border-red-800">
      ⚠️ {error}
    </div>
  );
}
