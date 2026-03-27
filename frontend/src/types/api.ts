export type DocumentStatus = 'uploaded' | 'processing' | 'ready' | 'failed';

export interface DocumentItem {
  id: string;
  filename: string;
  file_size_bytes: number;
  status: DocumentStatus;
  error_message?: string | null;
  page_count: number;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: DocumentItem[];
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  file_size_bytes: number;
  status: DocumentStatus;
  task_id?: string;
  message: string;
}

export interface DeleteDocumentsResponse {
  deleted_ids: string[];
  deleted_count: number;
}

export interface SourceChunk {
  document_id: string;
  filename: string;
  page_number: number;
  chunk_index: number;
  score: number;
  text_preview: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceChunk[];
  used_top_k: number;
  cached: boolean;
}

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatPayload {
  question: string;
  document_ids?: string[];
  top_k?: number;
  use_hybrid_search?: boolean;
  use_reranker?: boolean;
  history?: ChatTurn[];
}

export interface StreamEvent {
  event: 'chunk' | 'sources' | 'done';
  data: Record<string, unknown>;
}
