import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE_URL, askChat, deleteDocuments, listDocuments, streamChat, uploadPdf } from './lib/api';
import type { ChatPayload, ChatTurn, DocumentItem, SourceChunk } from './types/api';

type Role = 'user' | 'assistant';
type RequestAction = 'send' | 'regenerate';

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  sources: SourceChunk[];
  streaming?: boolean;
  failed?: boolean;
  sentAt?: number;
  regeneratedAt?: number;
  stoppedAt?: number;
  durationSeconds?: number;
}

const MAX_HISTORY_MESSAGES = 12;

const STATUS_CLASS: Record<DocumentItem['status'], string> = {
  uploaded: 'bg-sand text-ink dark:bg-amber-900/40 dark:text-amber-100',
  processing: 'bg-sky text-ink dark:bg-blue-900/40 dark:text-blue-100',
  ready: 'bg-mint text-ink dark:bg-emerald-900/40 dark:text-emerald-100',
  failed: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-200',
};

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

function parseSources(value: unknown): SourceChunk[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => {
      if (!item || typeof item !== 'object') {
        return null;
      }
      const row = item as Record<string, unknown>;
      return {
        document_id: String(row.document_id ?? ''),
        filename: String(row.filename ?? 'unknown.pdf'),
        page_number: Number(row.page_number ?? 0),
        chunk_index: Number(row.chunk_index ?? 0),
        score: Number(row.score ?? 0),
        text_preview: String(row.text_preview ?? ''),
      } satisfies SourceChunk;
    })
    .filter((item): item is SourceChunk => Boolean(item));
}

function buildHistoryPayload(messages: ChatMessage[]): ChatTurn[] {
  return messages
    .filter((message) => !message.failed && message.content.trim().length > 0)
    .slice(-MAX_HISTORY_MESSAGES)
    .map((message) => ({
      role: message.role,
      content: message.content.trim(),
    }));
}

function formatElapsedDuration(totalSeconds: number): string {
  const safeSeconds = Number.isFinite(totalSeconds) && totalSeconds > 0 ? Math.floor(totalSeconds) : 0;
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = safeSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function formatMinutesSeconds(totalSeconds: number): string {
  const safeSeconds = Number.isFinite(totalSeconds) && totalSeconds > 0 ? Math.floor(totalSeconds) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

function formatActionTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

function formatActionMeta(message: ChatMessage): string | null {
  const parts: string[] = [];

  if (typeof message.sentAt === 'number') {
    parts.push(`Send: ${formatActionTime(message.sentAt)}`);
  }
  if (typeof message.regeneratedAt === 'number') {
    parts.push(`Regenerate: ${formatActionTime(message.regeneratedAt)}`);
  }
  if (typeof message.stoppedAt === 'number') {
    parts.push(`Stop: ${formatActionTime(message.stoppedAt)}`);
  }

  return parts.length > 0 ? parts.join(' | ') : null;
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [lastUploadMessage, setLastUploadMessage] = useState<string>('');

  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [asking, setAsking] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [lastQuestion, setLastQuestion] = useState('');
  const [useStreaming, setUseStreaming] = useState(true);
  const [topK, setTopK] = useState(5);
  const [useHybrid, setUseHybrid] = useState(true);
  const [useReranker, setUseReranker] = useState(true);
  const [thinkingElapsedSeconds, setThinkingElapsedSeconds] = useState(0);

  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatFormRef = useRef<HTMLFormElement | null>(null);
  const activeAbortControllerRef = useRef<AbortController | null>(null);
  const generationStartedAtRef = useRef<number | null>(null);
  const stopRequestedAtRef = useRef<number | null>(null);

  const readyDocumentCount = useMemo(
    () => documents.filter((doc) => doc.status === 'ready').length,
    [documents],
  );
  const readyDocumentIdSet = useMemo(
    () => new Set(documents.filter((doc) => doc.status === 'ready').map((doc) => doc.id)),
    [documents],
  );
  const selectedReadyDocumentIds = useMemo(
    () => selectedDocumentIds.filter((id) => readyDocumentIdSet.has(id)),
    [selectedDocumentIds, readyDocumentIdSet],
  );
  const isMacOs = useMemo(() => {
    if (typeof navigator === 'undefined') {
      return false;
    }
    const platform = (navigator.platform || '').toLowerCase();
    const userAgent = (navigator.userAgent || '').toLowerCase();
    return platform.includes('mac') || userAgent.includes('mac os');
  }, []);
  const sendShortcutText = isMacOs ? '⌘⏎' : 'Ctrl+Enter';
  const latestUserQuestion = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'user') {
        return messages[i].content.trim();
      }
    }
    return '';
  }, [messages]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (typeof document === 'undefined' || typeof window === 'undefined') {
      return;
    }
    document.documentElement.classList.remove('dark');
    window.localStorage.removeItem('theme_mode');
  }, []);

  useEffect(() => {
    let cancelled = false;

    const fetchDocs = async () => {
      try {
        setLoadingDocs(true);
        const rows = await listDocuments();
        if (cancelled) {
          return;
        }
        setDocuments(rows);
        setSelectedDocumentIds((prev) => {
          const statusById = new Map(rows.map((d) => [d.id, d.status]));
          return prev.filter((id) => statusById.get(id) === 'ready');
        });
      } catch (error) {
        if (!cancelled) {
          setUploadError((error as Error).message);
        }
      } finally {
        if (!cancelled) {
          setLoadingDocs(false);
        }
      }
    };

    fetchDocs();
    const timer = setInterval(fetchDocs, 5000);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    return () => {
      activeAbortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!asking) {
      return;
    }

    const updateElapsed = () => {
      if (generationStartedAtRef.current === null) {
        setThinkingElapsedSeconds(0);
        return;
      }
      const elapsed = Math.floor((Date.now() - generationStartedAtRef.current) / 1000);
      setThinkingElapsedSeconds(Math.max(elapsed, 0));
    };

    updateElapsed();
    const timerId = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timerId);
  }, [asking]);

  const toggleSelection = (document: DocumentItem) => {
    if (document.status !== 'ready') {
      return;
    }
    setSelectedDocumentIds((prev) =>
      prev.includes(document.id)
        ? prev.filter((id) => id !== document.id)
        : [...prev, document.id],
    );
  };

  const onUpload = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setUploadError('Choose a PDF file first.');
      return;
    }

    setUploadError(null);
    setLastUploadMessage('');
    setUploading(true);

    try {
      const result = await uploadPdf(file);
      setLastUploadMessage(`${result.filename}: ${result.message}`);

      const rows = await listDocuments();
      setDocuments(rows);
      if (result.document_id && !selectedDocumentIds.includes(result.document_id)) {
        setSelectedDocumentIds((prev) => [...prev, result.document_id]);
      }

      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      setUploadError((error as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const onDeleteSelected = async () => {
    if (selectedDocumentIds.length === 0 || deleting) {
      return;
    }

    setUploadError(null);
    setDeleting(true);
    try {
      const result = await deleteDocuments(selectedDocumentIds);
      setLastUploadMessage(`Deleted ${result.deleted_count} document(s).`);
      const rows = await listDocuments();
      setDocuments(rows);
      setSelectedDocumentIds([]);
    } catch (error) {
      setUploadError((error as Error).message);
    } finally {
      setDeleting(false);
    }
  };

  const runQuestion = async (trimmedQuestion: string, action: RequestAction) => {
    if (!trimmedQuestion || asking) {
      return;
    }

    const skippedDocumentCount = selectedDocumentIds.length - selectedReadyDocumentIds.length;
    if (skippedDocumentCount > 0) {
      setChatError(
        `${skippedDocumentCount} selected document(s) are not ready yet and were ignored.`,
      );
    } else {
      setChatError(null);
    }

    const history = buildHistoryPayload(messages);
    const userMessage: ChatMessage = {
      id: makeId(),
      role: 'user',
      content: trimmedQuestion,
      sources: [],
    };

    const assistantId = makeId();
    const requestStartedAt = Date.now();
    const assistantPlaceholder: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
      streaming: useStreaming,
      sentAt: action === 'send' ? requestStartedAt : undefined,
      regeneratedAt: action === 'regenerate' ? requestStartedAt : undefined,
    };

    generationStartedAtRef.current = requestStartedAt;
    stopRequestedAtRef.current = null;
    setThinkingElapsedSeconds(0);
    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setAsking(true);
    setLastQuestion(trimmedQuestion);

    const payload: ChatPayload = {
      question: trimmedQuestion,
      top_k: topK,
      use_hybrid_search: useHybrid,
      use_reranker: useReranker,
      document_ids: selectedReadyDocumentIds.length > 0 ? selectedReadyDocumentIds : undefined,
      history,
    };

    const controller = new AbortController();
    activeAbortControllerRef.current = controller;

    try {
      if (useStreaming) {
        await streamChat(
          payload,
          (eventData) => {
            if (eventData.event === 'chunk') {
              const token = String(eventData.data.token ?? '');
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: `${msg.content}${token}` }
                    : msg,
                ),
              );
              return;
            }

            if (eventData.event === 'sources') {
              const sources = parseSources(eventData.data.sources);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId ? { ...msg, sources } : msg,
                ),
              );
              return;
            }

            if (eventData.event === 'done') {
              const doneDurationSeconds = Math.max(Math.floor((Date.now() - requestStartedAt) / 1000), 0);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, streaming: false, durationSeconds: doneDurationSeconds }
                    : msg,
                ),
              );
            }
          },
          controller.signal,
        );
      } else {
        const response = await askChat(payload, controller.signal);
        const responseDurationSeconds = Math.max(Math.floor((Date.now() - requestStartedAt) / 1000), 0);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: response.answer,
                  sources: response.sources,
                  streaming: false,
                  durationSeconds: responseDurationSeconds,
                }
              : msg,
          ),
        );
      }
    } catch (error) {
      const isAbort = error instanceof DOMException && error.name === 'AbortError';
      if (isAbort) {
        const stoppedAt = stopRequestedAtRef.current ?? Date.now();
        const elapsedAtStop = Math.max(Math.floor((stoppedAt - requestStartedAt) / 1000), 0);
        const stopDuration = formatElapsedDuration(elapsedAtStop);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: msg.content || `Generation stopped (${stopDuration}).`,
                  streaming: false,
                  stoppedAt,
                  durationSeconds: elapsedAtStop,
                }
              : msg,
          ),
        );
        return;
      }

      const message = (error as Error).message;
      const errorDurationSeconds = Math.max(Math.floor((Date.now() - requestStartedAt) / 1000), 0);
      setChatError(message);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: `Error: ${message}`,
                streaming: false,
                failed: true,
                durationSeconds: errorDurationSeconds,
              }
            : msg,
        ),
      );
    } finally {
      if (activeAbortControllerRef.current === controller) {
        activeAbortControllerRef.current = null;
      }
      generationStartedAtRef.current = null;
      stopRequestedAtRef.current = null;
      setAsking(false);
    }
  };

  const submitQuestion = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || asking) {
      return;
    }
    setQuestion('');
    await runQuestion(trimmed, 'send');
  };

  const onStopGeneration = () => {
    if (!asking) {
      return;
    }
    stopRequestedAtRef.current = Date.now();
    activeAbortControllerRef.current?.abort();
  };

  const onRegenerate = async () => {
    const prompt = latestUserQuestion || lastQuestion;
    if (!prompt || asking) {
      return;
    }
    await runQuestion(prompt, 'regenerate');
  };

  const onCopyMessage = async (messageId: string, content: string) => {
    const text = content.trim();
    if (!text || typeof navigator === 'undefined' || !navigator.clipboard) {
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId((current) => (current === messageId ? null : current)), 1200);
    } catch {
      // Ignore clipboard API errors.
    }
  };

  const onQuestionKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.nativeEvent.isComposing) {
      return;
    }
    const hasSendShortcut = event.metaKey || event.ctrlKey;
    if (!hasSendShortcut) {
      return;
    }
    event.preventDefault();
    if (asking || question.trim().length < 3) {
      return;
    }
    chatFormRef.current?.requestSubmit();
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-sand via-sky to-mint p-4 font-display text-ink transition-colors duration-300 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 dark:text-slate-100 md:p-8">
      <div className="pointer-events-none absolute -left-24 top-10 h-64 w-64 rounded-full bg-coral/20 blur-3xl dark:bg-coral/10" />
      <div className="pointer-events-none absolute right-0 top-1/2 h-80 w-80 -translate-y-1/2 rounded-full bg-sky/80 blur-3xl dark:bg-sky-500/20" />

      <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-6">
        <header className="rounded-3xl border border-white/60 bg-white/70 p-6 shadow-soft backdrop-blur transition-colors duration-300 dark:border-slate-700/60 dark:bg-slate-900/70">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
              PDF RAG Assistant
            </h1>
            <p className="mt-2 text-sm text-slate dark:text-slate-300">
              Upload documents, select one or multiple PDFs, and chat with grounded answers + citations.
            </p>
            <p className="mt-1 text-xs text-slate/80 dark:text-slate-400">API: {API_BASE_URL}</p>
          </div>
        </header>

        <main className="grid gap-6 lg:grid-cols-[360px_1fr]">
          <section className="rounded-3xl border border-white/60 bg-white/80 p-5 shadow-soft backdrop-blur transition-colors duration-300 dark:border-slate-700/60 dark:bg-slate-900/80">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Documents</h2>
              <span className="rounded-full bg-ink px-2.5 py-1 text-xs font-medium text-white dark:bg-slate-100 dark:text-slate-900">
                Ready: {readyDocumentCount}
              </span>
            </div>

            <form onSubmit={onUpload} className="space-y-3">
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                className="block w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm transition-colors dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
              />
              <button
                type="submit"
                disabled={uploading}
                className="w-full rounded-xl bg-ink px-4 py-2 text-sm font-semibold text-white transition hover:bg-ink/90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {uploading ? 'Uploading...' : 'Upload PDF'}
              </button>
              <button
                type="button"
                onClick={onDeleteSelected}
                disabled={deleting || selectedDocumentIds.length === 0}
                className="w-full rounded-xl bg-red-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {deleting ? 'Deleting...' : 'Delete'}
              </button>
            </form>

            {lastUploadMessage && (
              <p className="mt-3 rounded-xl bg-mint px-3 py-2 text-xs text-ink dark:bg-emerald-900/60 dark:text-emerald-100">
                {lastUploadMessage}
              </p>
            )}
            {uploadError && (
              <p className="mt-3 rounded-xl bg-red-100 px-3 py-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-200">
                {uploadError}
              </p>
            )}

            <div className="mt-5 flex items-center justify-between">
              <h3 className="text-sm font-semibold">Select Documents</h3>
              {loadingDocs && <span className="text-xs text-slate dark:text-slate-300">refreshing...</span>}
            </div>
            <p className="mt-1 text-xs text-slate dark:text-slate-300">Only documents with status "ready" are selectable for chat.</p>

            <div className="mt-3 max-h-[420px] space-y-2 overflow-auto pr-1">
              {documents.length === 0 && (
                <p className="rounded-xl bg-white px-3 py-3 text-xs text-slate dark:bg-slate-800 dark:text-slate-300">
                  No uploaded PDFs yet.
                </p>
              )}

              {documents.map((document) => (
                <label
                  key={document.id}
                  className={`block rounded-2xl border border-ink/10 bg-white p-3 transition dark:border-slate-700 dark:bg-slate-800 ${
                    document.status === 'ready'
                      ? 'cursor-pointer hover:border-ink/30 dark:hover:border-slate-500'
                      : 'cursor-not-allowed opacity-75'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={selectedDocumentIds.includes(document.id)}
                      onChange={() => toggleSelection(document)}
                      disabled={document.status !== 'ready'}
                      className="mt-1 h-4 w-4 rounded border-ink/50"
                    />

                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{document.filename}</p>
                      <p className="mt-1 text-xs text-slate dark:text-slate-300">
                        {formatFileSize(document.file_size_bytes)} | pages: {document.page_count} | chunks: {document.chunk_count}
                      </p>

                      <div className="mt-2 flex items-center gap-2">
                        <span
                          className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${STATUS_CLASS[document.status]}`}
                        >
                          {document.status}
                        </span>
                      </div>

                      {document.status === 'failed' && document.error_message && (
                        <p className="mt-2 rounded-lg bg-red-50 px-2 py-1 text-[11px] text-red-700 dark:bg-red-900/30 dark:text-red-200">
                          {document.error_message}
                        </p>
                      )}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </section>

          <section className="flex min-h-[72vh] flex-col rounded-3xl border border-white/60 bg-white/85 p-5 shadow-soft backdrop-blur transition-colors duration-300 dark:border-slate-700/60 dark:bg-slate-900/85">
            <div className="mb-4 flex flex-wrap items-center gap-3 rounded-2xl border border-ink/10 bg-white/80 p-3 text-sm transition-colors dark:border-slate-700 dark:bg-slate-800/80">
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={useStreaming}
                  onChange={(event) => setUseStreaming(event.target.checked)}
                  className="accent-coral"
                />
                Streaming
              </label>

              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={useHybrid}
                  onChange={(event) => setUseHybrid(event.target.checked)}
                  className="accent-coral"
                />
                Hybrid Search
              </label>

              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={useReranker}
                  onChange={(event) => setUseReranker(event.target.checked)}
                  className="accent-coral"
                />
                Reranker
              </label>

              <label className="inline-flex items-center gap-2">
                Top-K
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    if (!Number.isNaN(value) && value >= 1 && value <= 20) {
                      setTopK(value);
                    }
                  }}
                  className="w-16 rounded-md border border-ink/20 bg-white px-2 py-1 transition-colors dark:border-slate-600 dark:bg-slate-800"
                />
              </label>
            </div>

            <div className="flex-1 space-y-4 overflow-auto rounded-2xl border border-ink/10 bg-white/70 p-4 transition-colors dark:border-slate-700 dark:bg-slate-800/70">
              {messages.length === 0 && (
                <div className="rounded-2xl border border-dashed border-ink/20 bg-sky/30 p-4 text-sm text-slate dark:border-slate-600 dark:bg-slate-800 dark:text-slate-300">
                  Ask a question about your uploaded PDFs. If no document is selected, the backend searches all ready documents.
                </div>
              )}

              {messages.map((message) => {
                const actionMeta = message.role === 'assistant' ? formatActionMeta(message) : null;
                const requestDuration =
                  message.role === 'assistant' && typeof message.durationSeconds === 'number'
                    ? `Spent: ${formatMinutesSeconds(message.durationSeconds)}`
                    : null;

                return (
                  <article
                    key={message.id}
                    className={`max-w-3xl rounded-2xl px-4 py-3 text-sm shadow-sm ${
                      message.role === 'user'
                        ? 'ml-auto bg-ink text-white dark:bg-coral'
                        : message.failed
                          ? 'border border-red-200 bg-red-50 text-red-800 dark:border-red-700 dark:bg-red-900/30 dark:text-red-200'
                          : 'border border-ink/10 bg-white text-ink dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">
                      {message.content || (message.streaming ? `Thinking... ${formatElapsedDuration(thinkingElapsedSeconds)}` : '')}
                    </p>

                    {message.streaming && message.content.trim().length > 0 && (
                      <p className="mt-2 text-xs text-slate dark:text-slate-300">
                        Thinking... {formatElapsedDuration(thinkingElapsedSeconds)}
                      </p>
                    )}

                    {actionMeta && (
                      <p className="mt-2 text-xs text-slate dark:text-slate-300">
                        {actionMeta}
                      </p>
                    )}

                    {requestDuration && (
                      <p className="mt-2 text-xs text-slate dark:text-slate-300">
                        {requestDuration}
                      </p>
                    )}

                    {message.role === 'assistant' && message.content.trim().length > 0 && (
                      <div className="mt-2 flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => onCopyMessage(message.id, message.content)}
                          className="rounded-lg border border-ink/20 bg-white px-2.5 py-1 text-xs font-medium text-slate transition hover:bg-sky/20 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-700"
                        >
                          {copiedMessageId === message.id ? 'Copied' : 'Copy'}
                        </button>
                      </div>
                    )}

                    {message.role === 'assistant' && message.sources.length > 0 && (
                      <div className="mt-3 space-y-2 border-t border-ink/10 pt-3 dark:border-slate-700">
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate dark:text-slate-300">
                          Sources
                        </p>
                        {message.sources.map((source, index) => (
                          <div key={`${message.id}-${source.document_id}-${index}`} className="rounded-xl bg-sky/40 p-2 dark:bg-slate-700/70">
                            <p className="text-xs font-semibold">
                              {source.filename} | page {source.page_number} | score {source.score.toFixed(3)}
                            </p>
                            <p className="mt-1 text-xs text-slate dark:text-slate-300">{source.text_preview}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </article>
                );
              })}

              <div ref={chatEndRef} />
            </div>

            <form ref={chatFormRef} onSubmit={submitQuestion} className="mt-4 flex flex-col gap-3 sm:flex-row">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={onQuestionKeyDown}
                rows={2}
                placeholder="Ask something grounded in your PDFs..."
                className="min-h-[80px] flex-1 resize-y rounded-2xl border border-ink/20 bg-white px-4 py-3 text-sm outline-none transition focus:border-ink dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:placeholder:text-slate-400 dark:focus:border-slate-400"
              />
              <div className="flex flex-wrap items-center gap-2 self-end sm:self-auto">
                <div className="group relative inline-flex" title={`Shortcut: ${sendShortcutText}`}>
                  <button
                    type="submit"
                    disabled={asking || question.trim().length < 3}
                    aria-label={`Send. Shortcut: ${sendShortcutText}`}
                    className="rounded-2xl bg-coral px-5 py-3 text-sm font-semibold text-white transition hover:bg-coral/90 dark:bg-coral/90 dark:hover:bg-coral disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {asking ? 'Generating...' : 'Send'}
                  </button>
                  <span className="pointer-events-none absolute -top-9 left-1/2 z-10 -translate-x-1/2 whitespace-nowrap rounded-lg bg-ink px-2 py-1 text-[11px] font-medium text-white opacity-0 shadow-sm transition group-hover:opacity-100 dark:bg-slate-100 dark:text-slate-900">
                    {sendShortcutText}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={onStopGeneration}
                  disabled={!asking}
                  className="rounded-2xl border border-ink/20 bg-white px-4 py-3 text-sm font-semibold text-ink transition hover:bg-sky/20 disabled:cursor-not-allowed disabled:opacity-60 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700"
                >
                  Stop
                </button>
                <button
                  type="button"
                  onClick={onRegenerate}
                  disabled={asking || !(latestUserQuestion || lastQuestion)}
                  className="rounded-2xl border border-ink/20 bg-white px-4 py-3 text-sm font-semibold text-ink transition hover:bg-sky/20 disabled:cursor-not-allowed disabled:opacity-60 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700"
                >
                  Regenerate
                </button>
              </div>
            </form>

            {chatError && (
              <p className="mt-3 rounded-xl bg-red-100 px-3 py-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-200">
                {chatError}
              </p>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
