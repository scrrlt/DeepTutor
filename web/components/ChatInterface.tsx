import React from 'react';
import {
  Bot,
  User,
  Database,
  Globe,
  Trash2,
  BookOpen,
  ExternalLink,
  Loader2,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { processLatexContent } from '@/lib/latex';

interface Message {
  role: string;
  content: string;
  isStreaming?: boolean;
  sources?: {
    rag?: Array<{ kb_name: string }>;
    web?: Array<{ url: string; title?: string }>;
  };
}

interface ChatInterfaceProps {
  messages: Message[];
  chatState: any;
  setChatState: any;
  newChatSession: () => void;
  inputMessage: string;
  setInputMessage: (value: string) => void;
  handleSend: () => void;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  isLoading: boolean;
  t: (key: string) => string;
  inputRef: React.RefObject<HTMLInputElement>;
  chatEndRef: React.RefObject<HTMLDivElement>;
}

export function ChatInterface({
  messages,
  chatState,
  setChatState,
  newChatSession,
  inputMessage,
  setInputMessage,
  handleSend,
  handleKeyDown,
  isLoading,
  t,
  inputRef,
  chatEndRef,
}: ChatInterfaceProps) {
  return (
    <>
      {/* Header Bar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          {/* Mode Toggles */}
          <button
            onClick={() =>
              setChatState((prev: any) => ({
                ...prev,
                enableRag: !prev.enableRag,
              }))
            }
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
              chatState.enableRag
                ? "bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300"
                : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
            }`}
          >
            <Database className="w-3 h-3" />
            {t("RAG")}
          </button>

          <button
            onClick={() =>
              setChatState((prev: any) => ({
                ...prev,
                enableWebSearch: !prev.enableWebSearch,
              }))
            }
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
              chatState.enableWebSearch
                ? "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300"
                : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
            }`}
          >
            <Globe className="w-3 h-3" />
            {t("Web Search")}
          </button>
        </div>

        <button
          onClick={newChatSession}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-500 dark:text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-lg transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
          {t("New Chat")}
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className="flex gap-4 w-full max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-2"
          >
            {msg.role === "user" ? (
              <>
                <div className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center shrink-0">
                  <User className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                </div>
                <div className="flex-1 bg-slate-100 dark:bg-slate-700 px-4 py-3 rounded-2xl rounded-tl-none text-slate-800 dark:text-slate-200">
                  {msg.content}
                </div>
              </>
            ) : (
              <>
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shrink-0 shadow-lg shadow-blue-500/30">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="flex-1 space-y-3">
                  <div className="bg-white dark:bg-slate-800 px-5 py-4 rounded-2xl rounded-tl-none border border-slate-200 dark:border-slate-700 shadow-sm">
                    <div className="prose prose-slate dark:prose-invert prose-sm max-w-none">
                      <ReactMarkdown
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {processLatexContent(msg.content)}
                      </ReactMarkdown>
                    </div>

                    {/* Loading indicator */}
                    {msg.isStreaming && (
                      <div className="flex items-center gap-2 mt-3 text-blue-600 dark:text-blue-400 text-sm">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>{t("Generating response...")}</span>
                      </div>
                    )}
                  </div>

                  {/* Sources */}
                  {msg.sources &&
                    (msg.sources.rag?.length ?? 0) +
                      (msg.sources.web?.length ?? 0) >
                      0 && (
                      <div className="flex flex-wrap gap-2">
                        {msg.sources.rag?.map((source, i) => (
                          <div
                            key={`rag-${i}`}
                            className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-lg text-xs"
                          >
                            <BookOpen className="w-3 h-3" />
                            <span>{source.kb_name}</span>
                          </div>
                        ))}
                        {msg.sources.web?.slice(0, 3).map((source, i) => (
                          <a
                            key={`web-${i}`}
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg text-xs hover:bg-emerald-100 dark:hover:bg-emerald-900/50 transition-colors"
                          >
                            <Globe className="w-3 h-3" />
                            <span className="max-w-[150px] truncate">
                              {source.title || source.url}
                            </span>
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        ))}
                      </div>
                    )}
                </div>
              </>
            )}
          </div>
        ))}

        {/* Status indicator */}
        {isLoading && chatState.currentStage && (
          <div className="flex gap-4 w-full max-w-4xl mx-auto">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shrink-0">
              <Loader2 className="w-4 h-4 text-white animate-spin" />
            </div>
            <div className="flex-1 bg-slate-100 dark:bg-slate-800 px-4 py-3 rounded-2xl rounded-tl-none">
              <div className="flex items-center gap-2 text-slate-600 dark:text-slate-300 text-sm">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                </span>
                {chatState.currentStage === "rag" &&
                  t("Searching knowledge base...")}
                {chatState.currentStage === "web" &&
                  t("Searching the web...")}
                {chatState.currentStage === "generating" &&
                  t("Generating response...")}
                {!["rag", "web", "generating"].includes(
                  chatState.currentStage,
                ) && chatState.currentStage}
              </div>
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Input Area - Fixed at bottom */}
      <div className="border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-6 py-4">
        <div className="max-w-4xl mx-auto relative">
          <input
            ref={inputRef}
            type="text"
            className="w-full px-5 py-3.5 pr-14 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all placeholder:text-slate-400 dark:placeholder:text-slate-500 text-slate-700 dark:text-slate-200"
            placeholder={t("Type your message...")}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !inputMessage.trim()}
            className="absolute right-2 top-2 bottom-2 aspect-square bg-blue-600 text-white rounded-lg flex items-center justify-center hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-all"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </>
  );
}
