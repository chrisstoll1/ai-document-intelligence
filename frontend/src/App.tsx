import DeleteOutlineRounded from '@mui/icons-material/DeleteOutlineRounded'
import DescriptionOutlined from '@mui/icons-material/DescriptionOutlined'
import FileUploadOutlined from '@mui/icons-material/FileUploadOutlined'
import FolderOpenOutlined from '@mui/icons-material/FolderOpenOutlined'
import LocalLibraryOutlined from '@mui/icons-material/LocalLibraryOutlined'
import RestartAltRounded from '@mui/icons-material/RestartAltRounded'
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  IconButton,
  Paper,
  Stack,
  Typography,
} from '@mui/material'
import { ChangeEvent, useEffect, useState } from 'react'

import {
  deleteDocument,
  DocumentRecord,
  getDocumentMetadata,
  listDocuments,
  resetDocuments,
  uploadDocument,
} from './api'

const statusColors: Record<string, 'success' | 'error' | 'warning' | 'default'> = {
  ready: 'success',
  indexed_lexical: 'success',
  failed: 'error',
  index_failed: 'warning',
  processing: 'warning',
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [busyDocument, setBusyDocument] = useState<string | null>(null)
  const [metadataSummary, setMetadataSummary] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    listDocuments()
      .then((records) => {
        if (active) setDocuments(records)
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : 'Could not load documents')
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const uploaded = await uploadDocument(file)
      setDocuments((current) => [uploaded, ...current.filter((item) => item.id !== uploaded.id)])
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(document: DocumentRecord) {
    if (!window.confirm(`Remove ${document.filename} and its indexes?`)) return
    setBusyDocument(document.id)
    setError(null)
    try {
      await deleteDocument(document.id)
      setDocuments((current) => current.filter((item) => item.id !== document.id))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Delete failed')
    } finally {
      setBusyDocument(null)
    }
  }

  async function handleReset() {
    if (!window.confirm('Remove every document and rebuild the local collection?')) return
    setError(null)
    try {
      await resetDocuments()
      setDocuments([])
      setMetadataSummary({})
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Collection reset failed')
    }
  }

  async function inspectMetadata(document: DocumentRecord) {
    setBusyDocument(document.id)
    setError(null)
    try {
      const metadata = await getDocumentMetadata(document.id)
      const counts = new Map<string, number>()
      for (const entity of metadata.entities) counts.set(entity.label, (counts.get(entity.label) ?? 0) + 1)
      const summary = counts.size
        ? [...counts].map(([label, count]) => `${count} ${label.toLowerCase()}`).join(' · ')
        : 'No selected entities detected'
      setMetadataSummary((current) => ({ ...current, [document.id]: summary }))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Metadata could not be loaded')
    } finally {
      setBusyDocument(null)
    }
  }

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', color: 'text.primary', pb: 8 }}>
      <Box component="header" sx={{ borderBottom: '1px solid', borderColor: 'divider', bgcolor: '#0f2632' }}>
        <Container maxWidth="xl" sx={{ py: { xs: 3, md: 4 } }}>
          <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" gap={3}>
            <Stack direction="row" spacing={2} alignItems="center">
              <Box sx={{ width: 46, height: 46, border: '1px solid #7bb7a7', display: 'grid', placeItems: 'center' }}>
                <LocalLibraryOutlined sx={{ color: '#b7e4d7' }} />
              </Box>
              <Box>
                <Typography variant="overline" sx={{ color: '#91c8ba', letterSpacing: '0.18em' }}>
                  Local evidence workspace
                </Typography>
                <Typography variant="h4" sx={{ color: '#f5f0e8', fontFamily: 'Georgia, serif' }}>
                  Document Intelligence
                </Typography>
              </Box>
            </Stack>
            <Stack direction="row" spacing={1.5} alignItems="center">
              <Button
                component="label"
                variant="contained"
                startIcon={uploading ? <CircularProgress size={17} color="inherit" /> : <FileUploadOutlined />}
                disabled={uploading}
              >
                {uploading ? 'Processing PDF' : 'Add PDF'}
                <input hidden type="file" accept="application/pdf,.pdf" onChange={handleUpload} />
              </Button>
              <Button
                variant="outlined"
                color="inherit"
                startIcon={<RestartAltRounded />}
                disabled={!documents.length}
                onClick={handleReset}
                sx={{ color: '#f5f0e8', borderColor: '#8da0a8' }}
              >
                Reset
              </Button>
            </Stack>
          </Stack>
        </Container>
      </Box>

      <Container maxWidth="xl" sx={{ mt: 4 }}>
        {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
        <Stack direction={{ xs: 'column', lg: 'row' }} spacing={3} alignItems="flex-start">
          <Paper variant="outlined" sx={{ width: { xs: '100%', lg: 390 }, overflow: 'hidden' }}>
            <Box sx={{ p: 2.5 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                <Typography variant="h6" sx={{ fontFamily: 'Georgia, serif' }}>Collection</Typography>
                <Typography variant="caption" color="text.secondary">
                  {documents.length} {documents.length === 1 ? 'document' : 'documents'}
                </Typography>
              </Stack>
            </Box>
            <Divider />
            {loading ? (
              <Box sx={{ p: 6, display: 'grid', placeItems: 'center' }}><CircularProgress size={28} /></Box>
            ) : documents.length === 0 ? (
              <Stack sx={{ p: 5, textAlign: 'center' }} spacing={1.5} alignItems="center">
                <FolderOpenOutlined sx={{ fontSize: 40, color: 'text.disabled' }} />
                <Typography variant="subtitle1">No documents yet</Typography>
                <Typography variant="body2" color="text.secondary">
                  Add a digital or scanned PDF to create the local evidence index.
                </Typography>
              </Stack>
            ) : (
              <Stack divider={<Divider flexItem />}>
                {documents.map((document) => (
                  <Box key={document.id} sx={{ p: 2.25 }}>
                    <Stack direction="row" spacing={1.5} alignItems="flex-start">
                      <DescriptionOutlined sx={{ mt: 0.25, color: 'primary.main' }} />
                      <Box sx={{ minWidth: 0, flex: 1 }}>
                        <Typography fontWeight={650} noWrap title={document.filename}>{document.filename}</Typography>
                        <Stack direction="row" spacing={1} sx={{ mt: 1, mb: 1.25 }} alignItems="center">
                          <Chip
                            size="small"
                            label={document.status.replace('_', ' ')}
                            color={statusColors[document.status] ?? 'default'}
                          />
                          <Typography variant="caption" color="text.secondary">{formatBytes(document.size_bytes)}</Typography>
                        </Stack>
                        {(document.error_message || document.metadata_error) && (
                          <Typography variant="caption" color="error.main" display="block" sx={{ mb: 1 }}>
                            {document.error_message ?? document.metadata_error}
                          </Typography>
                        )}
                        {metadataSummary[document.id] && (
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                            {metadataSummary[document.id]}
                          </Typography>
                        )}
                        <Button size="small" onClick={() => inspectMetadata(document)} disabled={busyDocument === document.id}>
                          Inspect metadata
                        </Button>
                      </Box>
                      <IconButton
                        aria-label={`Delete ${document.filename}`}
                        size="small"
                        onClick={() => handleDelete(document)}
                        disabled={busyDocument === document.id}
                      >
                        <DeleteOutlineRounded fontSize="small" />
                      </IconButton>
                    </Stack>
                  </Box>
                ))}
              </Stack>
            )}
          </Paper>

          <Paper
            variant="outlined"
            sx={{
              flex: 1,
              width: '100%',
              minHeight: 480,
              p: { xs: 3, md: 5 },
              backgroundImage: 'linear-gradient(rgba(24, 64, 78, 0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(24, 64, 78, 0.035) 1px, transparent 1px)',
              backgroundSize: '28px 28px',
            }}
          >
            <Typography variant="overline" color="primary.main" sx={{ letterSpacing: '0.16em' }}>Evidence desk</Typography>
            <Typography variant="h3" sx={{ maxWidth: 720, mt: 1, mb: 2, fontFamily: 'Georgia, serif' }}>
              Ask the collection, then inspect the source.
            </Typography>
            <Typography color="text.secondary" sx={{ maxWidth: 660, lineHeight: 1.75 }}>
              Hybrid retrieval and grounded answers will appear here. Every citation remains connected to its stored passage and original PDF page.
            </Typography>
            <Divider sx={{ my: 4 }} />
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={3}>
              <Box><Typography variant="h5" fontFamily="Georgia, serif">01</Typography><Typography variant="body2">Upload and index locally</Typography></Box>
              <Box><Typography variant="h5" fontFamily="Georgia, serif">02</Typography><Typography variant="body2">Retrieve lexical + semantic evidence</Typography></Box>
              <Box><Typography variant="h5" fontFamily="Georgia, serif">03</Typography><Typography variant="body2">Trace claims back to pages</Typography></Box>
            </Stack>
          </Paper>
        </Stack>
      </Container>
    </Box>
  )
}
