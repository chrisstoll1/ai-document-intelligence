export type DocumentRecord = {
  id: string
  filename: string
  size_bytes: number
  status: string
  error_message: string | null
  embedding_model: string | null
  metadata_status: string
  metadata_model: string | null
  metadata_error: string | null
}

export type EntityMention = {
  page_number: number
  label: string
  text: string
  normalized_text: string
  char_start: number
  char_end: number
  confidence: number | null
}

export type DocumentMetadata = {
  document_id: string
  status: string
  model: string | null
  error_message: string | null
  entities: EntityMention[]
}

function errorDetail(payload: unknown, fallback: string) {
  if (typeof payload === 'object' && payload !== null && 'detail' in payload) {
    const detail = payload.detail
    if (typeof detail === 'string') return detail
  }
  return fallback
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    let payload: unknown
    try {
      payload = await response.json()
    } catch {
      payload = undefined
    }
    throw new Error(errorDetail(payload, `Request failed with status ${response.status}`))
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export function listDocuments() {
  return request<DocumentRecord[]>('/api/documents')
}

export function uploadDocument(file: File) {
  const body = new FormData()
  body.append('upload', file)
  return request<DocumentRecord>('/api/documents', { method: 'POST', body })
}

export function deleteDocument(documentId: string) {
  return request<void>(`/api/documents/${documentId}`, { method: 'DELETE' })
}

export function resetDocuments() {
  return request<{ deleted_count: number }>('/api/documents', { method: 'DELETE' })
}

export function getDocumentMetadata(documentId: string) {
  return request<DocumentMetadata>(`/api/documents/${documentId}/metadata`)
}
