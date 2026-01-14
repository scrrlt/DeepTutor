import React from 'react';
import Link from 'next/link';
import {
  Calculator,
  PenTool,
  Microscope,
  Lightbulb,
  GraduationCap,
  Edit3,
  Database,
  Globe,
  Send,
  Loader2,
} from 'lucide-react';

interface KnowledgeBase {
  name: string;
  is_default?: boolean;
}

interface WelcomeScreenProps {
  inputMessage: string;
  setInputMessage: (value: string) => void;
  handleSend: () => void;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  chatState: any;
  setChatState: any;
  kbs: KnowledgeBase[];
  kbError: string | null;
  isLoading: boolean;
  t: (key: string) => string;
  inputRef: React.RefObject<HTMLInputElement>;
}

export function WelcomeScreen({
  inputMessage,
  setInputMessage,
  handleSend,
  handleKeyDown,
  chatState,
  setChatState,
  kbs,
  kbError,
  isLoading,
  t,
  inputRef,
}: WelcomeScreenProps) {
  const quickActions = [
    {
      icon: Calculator,
      label: t("Smart Problem Solving"),
      href: "/solver",
      color: "blue",
      description: "Multi-agent reasoning",
    },
    {
      icon: PenTool,
      label: t("Generate Practice Questions"),
      href: "/question",
      color: "purple",
      description: "Auto-validated quizzes",
    },
    {
      icon: Microscope,
      label: t("Deep Research Reports"),
      href: "/research",
      color: "emerald",
      description: "Comprehensive analysis",
    },
    {
      icon: Lightbulb,
      label: t("Generate Novel Ideas"),
      href: "/ideagen",
      color: "amber",
      description: "Brainstorm & synthesize",
    },
    {
      icon: GraduationCap,
      label: t("Guided Learning"),
      href: "/guide",
      color: "indigo",
      description: "Step-by-step tutoring",
    },
    {
      icon: Edit3,
      label: t("Co-Writer"),
      href: "/co_writer",
      color: "pink",
      description: "Collaborative writing",
    },
  ];

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6">
      <div className="text-center max-w-2xl mx-auto mb-8">
        <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-3 tracking-tight">
          {t("Welcome to DeepTutor")}
        </h1>
        <p className="text-lg text-slate-500 dark:text-slate-400">
          {t("How can I help you today?")}
        </p>
      </div>

      {/* Input Box - Centered */}
      <div className="w-full max-w-2xl mx-auto mb-12">
        {/* Mode Toggles */}
        <div className="flex items-center justify-between mb-3 px-1">
          <div className="flex items-center gap-2">
            {/* RAG Toggle */}
            <button
              onClick={() =>
                setChatState((prev: any) => ({
                  ...prev,
                  enableRag: !prev.enableRag,
                }))
              }
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                chatState.enableRag
                  ? "bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-700"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700"
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              {t("RAG")}
            </button>

            {/* Web Search Toggle */}
            <button
              onClick={() =>
                setChatState((prev: any) => ({
                  ...prev,
                  enableWebSearch: !prev.enableWebSearch,
                }))
              }
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                chatState.enableWebSearch
                  ? "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700"
              }`}
            >
              <Globe className="w-3.5 h-3.5" />
              {t("Web Search")}
            </button>
          </div>

          {/* KB Selector */}
          {chatState.enableRag && (
            <div className="flex flex-col gap-2">
              <select
                value={chatState.selectedKb}
                onChange={(e) =>
                  setChatState((prev: any) => ({
                    ...prev,
                    selectedKb: e.target.value,
                  }))
                }
                className="text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 outline-none focus:border-blue-400 dark:text-slate-200"
              >
                {kbs.map((kb) => (
                  <option key={kb.name} value={kb.name}>
                    {kb.name}
                  </option>
                ))}
              </select>
              {kbError && (
                <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded border border-red-200 dark:border-red-800">
                  ⚠️ {kbError}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input Field */}
        <div className="relative">
          <input
            ref={inputRef}
            type="text"
            className="w-full px-5 py-4 pr-14 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all placeholder:text-slate-400 dark:placeholder:text-slate-500 text-slate-700 dark:text-slate-200 shadow-lg shadow-slate-200/50 dark:shadow-slate-900/50"
            placeholder={t("Ask anything...")}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !inputMessage.trim()}
            className="absolute right-2 top-2 bottom-2 aspect-square bg-blue-600 text-white rounded-xl flex items-center justify-center hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-all shadow-md shadow-blue-500/20"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Quick Actions Grid */}
      <div className="w-full max-w-3xl mx-auto">
        <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-4 text-center">
          {t("Explore Modules")}
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {quickActions.map((action, i) => (
            <Link
              key={i}
              href={action.href}
              className={`group p-4 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:shadow-lg hover:border-${action.color}-300 dark:hover:border-${action.color}-600 transition-all`}
            >
              <div
                className={`w-10 h-10 rounded-xl bg-${action.color}-100 dark:bg-${action.color}-900/30 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform`}
              >
                <action.icon
                  className={`w-5 h-5 text-${action.color}-600 dark:text-${action.color}-400`}
                />
              </div>
              <h4 className="font-semibold text-slate-900 dark:text-slate-100 text-sm mb-1">
                {action.label}
              </h4>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {action.description}
              </p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
