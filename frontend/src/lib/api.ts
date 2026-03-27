import type {
  ChatPayload,
  ChatResponse,
  DeleteDocumentsResponse,
  DocumentItem,
  DocumentListResponse,
  StreamEvent,
  UploadResponse,
} from '../types/api';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || 'http://localhost:18000/api/v1';

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === 'string') {
        message = body.detail;
      }
    } catch {
      // Ignore non-JSON errors.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  return parseJsonOrThrow<UploadResponse>(response);
}

export async function listDocuments(): Promise<DocumentItem[]> {
  const response = await fetch(`${API_BASE_URL}/documents?limit=200`);
  const data = await parseJsonOrThrow<DocumentListResponse>(response);
  return data.items;
}

export async function deleteDocuments(documentIds: string[]): Promise<DeleteDocumentsResponse> {
  const response = await fetch(`${API_BASE_URL}/documents/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_ids: documentIds }),
  });

  return parseJsonOrThrow<DeleteDocumentsResponse>(response);
}

export async function askChat(payload: ChatPayload, signal?: AbortSignal): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  return parseJsonOrThrow<ChatResponse>(response);
}

function parseSseEvent(rawEvent: string): StreamEvent | null {
  const lines = rawEvent.split('\n');
  let event: StreamEvent['event'] | null = null;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith('event:')) {
      const value = line.slice('event:'.length).trim();
      if (value === 'chunk' || value === 'sources' || value === 'done') {
        event = value;
      }
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trim());
    }
  }

  if (!event) {
    return null;
  }

  const dataRaw = dataLines.join('\n') || '{}';
  let data: Record<string, unknown> = {};

  try {
    data = JSON.parse(dataRaw) as Record<string, unknown>;
  } catch {
    data = {};
  }

  return { event, data };
}

export async function streamChat(
  payload: ChatPayload,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    let message = `Streaming failed: ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === 'string') {
        message = body.detail;
      }
    } catch {
      // Ignore non-JSON errors.
    }
    throw new Error(message);
  }

  if (!response.body) {
    throw new Error('Streaming failed: empty response body');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    const packets = buffer.split('\n\n');
    buffer = packets.pop() || '';

    for (const packet of packets) {
      const event = parseSseEvent(packet);
      if (event) {
        onEvent(event);
      }
    }
  }

  if (buffer.trim().length > 0) {
    const event = parseSseEvent(buffer);
    if (event) {
      onEvent(event);
    }
  }
}

export { API_BASE_URL };
