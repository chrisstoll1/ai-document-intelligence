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
})
