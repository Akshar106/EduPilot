// Shapes of the FastAPI responses this app reads. Field names match the API.

export interface DomainInfo {
  name: string;
  abbr: string;
  color: string;
  description: string;
}

export interface AppConfig {
  available_models: string[];
  groq_models: string[];
  default_model: string;
  domains: Record<string, DomainInfo>;
  defaults: { top_k: number; rerank_top_k: number; max_query_chars: number };
}

export interface Source {
  source_num: number;
  domain: string;
  citation_label: string;
  filename?: string | null;
  page_number?: number | null;
  section_path?: string | null;
  relevance?: number;
  text?: string;
  kind?: string;
  /** "book" for a textbook excerpt; course material otherwise. */
  source_type?: string;
}

export interface QuizItem {
  question: string;
  options: string[];
  answer_index: number;
  explanation: string;
  source: number | null;
}

export interface FlashcardItem {
  front: string;
  back: string;
  source: number | null;
}

export type Practice =
  | { type: "quiz"; title: string; items: QuizItem[] }
  | { type: "flashcards"; title: string; items: FlashcardItem[] };

export type Task = "answer" | "quiz" | "flashcards";

export interface TraceInfo {
  run_id: string | null;
  traced: boolean;
}

/** The `debug` block: whatever the pipeline recorded about the turn. */
export interface Diagnostics {
  router?: { intent_type: string; domains: string[]; reasoning?: string };
  sub_questions?: { question: string; domain: string }[];
  retrieval?: {
    domain: string;
    chunks: number;
    grades?: { relevant: number; total: number }[];
    rewrites?: string[];
    calculations?: string[];
  }[];
  verifier?: VerifierScores;
  verification?: VerifierScores;
  audits?: { grounding_score: number | null }[];
  profile?: { weak_topics?: string[] };
  usage?: { calls?: number; input_tokens?: number; output_tokens?: number; fallbacks?: number };
  cache?: { hit: boolean; similarity?: number };
  trace?: TraceInfo;
  total_latency_ms?: number;
  practice?: Practice | null;
  [key: string]: unknown;
}

export interface VerifierScores {
  skipped?: boolean;
  grounding_score?: number | null;
  coverage_score?: number | null;
  quality_score?: number | null;
}

export interface ChatResult {
  session_id: string;
  final_answer: string;
  intent_type?: string;
  detected_domains?: string[];
  is_course_related?: boolean;
  needs_clarification?: boolean;
  awaiting_clarification?: boolean;
  refused?: boolean;
  task?: Task;
  grounding_score?: number | null;
  guardrail_action?: string | null;
  verification_revised?: boolean;
  sources: Source[];
  practice?: Practice | null;
  debug?: Diagnostics;
  assistant_message_id?: number;
  user_message_id?: number;
}

export interface SessionSummary {
  session_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface StoredMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  detected_domains?: string[] | null;
  quality_score?: number | null;
  pipeline_meta?: Diagnostics | null;
}

export interface StudySessionSummary {
  ss_session_id: string;
  name: string;
  description: string | null;
  updated_at: string;
  doc_count: number;
  total_chunks: number;
}

export interface StudyDocument {
  id: number;
  filename: string;
  file_type: string | null;
  chunk_count: number;
  file_size_bytes: number | null;
  /** Where the ingestion worker is with it. */
  status?: "queued" | "indexing" | "ready" | "failed";
  error?: string | null;
}

export interface StudyResult {
  final_answer: string;
  refused?: boolean;
  task?: Task;
  practice?: Practice | null;
  grounding_score?: number | null;
  sources: { citation_label: string; filename: string | null; page_number: number | null }[];
  debug?: Diagnostics;
  assistant_message_id?: number;
}

/** One file accepted for indexing; its job reports progress. */
export interface QueuedUpload {
  filename: string;
  job_id: number;
  document_id?: number | null;
  status: string;
  warnings?: string[];
}

export interface Job {
  job_id: number;
  status: "queued" | "running" | "done" | "failed";
  filename: string;
  chunks_indexed: number | null;
  error: string | null;
}

export interface ProfileTopic {
  name: string;
  domain: string;
  asks: number;
  confusion: number;
  last_seen: number;
}

export interface Profile {
  topics: ProfileTopic[];
  weak_topics: ProfileTopic[];
  depth: string | null;
  turns: number;
  updated_at: number | null;
}
