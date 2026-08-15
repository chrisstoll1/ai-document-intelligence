import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import type { DocumentRecord } from './api'

const readyDocument: DocumentRecord = {
  id: 'a'.repeat(64),
  filename: 'annual-report.pdf',
  size_bytes: 2048,
  status: 'ready',
  error_message: null,
  embedding_model: 'mini-lm',
  metadata_status: 'ready',
  metadata_model: 'spacy',
  metadata_error: null,
}

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('document collection', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('shows the empty collection state', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse([]))

    render(<App />)

    expect(await screen.findByText('No documents yet')).toBeInTheDocument()
    expect(screen.getByText('0 documents')).toBeInTheDocument()
  })

  it('uploads a PDF and adds it to the collection', async () => {
    const user = userEvent.setup()
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse(readyDocument))
    render(<App />)
    await screen.findByText('No documents yet')

    const file = new File(['%PDF-1.7'], 'annual-report.pdf', { type: 'application/pdf' })
    await user.upload(screen.getByLabelText('Add PDF'), file)

    expect(await screen.findByText('annual-report.pdf')).toBeInTheDocument()
    expect(screen.getByText('ready')).toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls[1][1]).toMatchObject({ method: 'POST' })
  })

  it('deletes a document after confirmation', async () => {
    const user = userEvent.setup()
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse([readyDocument]))
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
    render(<App />)
    await screen.findByText('annual-report.pdf')

    await user.click(screen.getByRole('button', { name: 'Delete annual-report.pdf' }))

    await waitFor(() => expect(screen.queryByText('annual-report.pdf')).not.toBeInTheDocument())
    expect(screen.getByText('No documents yet')).toBeInTheDocument()
  })

  it('renders grounded claims, evidence, and source-page links', async () => {
    const user = userEvent.setup()
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse([readyDocument]))
      .mockResolvedValueOnce(jsonResponse({
        status: 'answered',
        answer: 'Revenue increased.',
        claims: [{ text: 'Revenue increased.', citation_ids: ['C1'] }],
        contexts: [{
          context_id: 'C1',
          chunk_id: 'b'.repeat(64),
          document_id: readyDocument.id,
          document_name: readyDocument.filename,
          text: 'Revenue increased from 2018 to 2019.',
          page_start: 2,
          page_end: 2,
          score: 0.01,
          keyword_rank: 1,
          semantic_rank: 2,
        }],
        failure_reason: null,
      }))
    render(<App />)
    await screen.findByText('annual-report.pdf')

    await user.type(screen.getByLabelText('Ask a question'), 'How did revenue change?')
    await user.click(screen.getByRole('button', { name: 'Find evidence' }))

    expect(await screen.findByText('Revenue increased.')).toBeInTheDocument()
    expect(screen.getByText('Revenue increased from 2018 to 2019.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open source' })).toHaveAttribute(
      'href',
      `/api/documents/${readyDocument.id}/pdf#page=2`,
    )
    await user.click(screen.getByRole('button', { name: 'C1' }))
  })

  it('shows the insufficient-evidence state', async () => {
    const user = userEvent.setup()
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse([readyDocument]))
      .mockResolvedValueOnce(jsonResponse({
        status: 'insufficient_evidence',
        answer: 'Insufficient evidence in the retrieved passages.',
        claims: [],
        contexts: [],
        failure_reason: null,
      }))
    render(<App />)
    await screen.findByText('annual-report.pdf')

    await user.type(screen.getByLabelText('Ask a question'), 'What is not in the collection?')
    await user.click(screen.getByRole('button', { name: 'Find evidence' }))

    expect(await screen.findByText(/did not provide enough evidence/)).toBeInTheDocument()
  })
})
