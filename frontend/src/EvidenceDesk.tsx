import AutoAwesomeOutlined from '@mui/icons-material/AutoAwesomeOutlined'
import FindInPageOutlined from '@mui/icons-material/FindInPageOutlined'
import OpenInNewRounded from '@mui/icons-material/OpenInNewRounded'
import SearchRounded from '@mui/icons-material/SearchRounded'
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { FormEvent, useState } from 'react'

import { answerQuestion, AnswerResult, GroundingContext } from './api'

type EvidenceDeskProps = {
  enabled: boolean
}

function pageLabel(context: GroundingContext) {
  return context.page_start === context.page_end
    ? `Page ${context.page_start}`
    : `Pages ${context.page_start}-${context.page_end}`
}

function sourceUrl(context: GroundingContext) {
  return `/api/documents/${context.document_id}/pdf#page=${context.page_start}`
}

export default function EvidenceDesk({ enabled }: EvidenceDeskProps) {
  const [query, setQuery] = useState('')
  const [answering, setAnswering] = useState(false)
  const [result, setResult] = useState<AnswerResult | null>(null)
  const [focusedContext, setFocusedContext] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function submitQuestion(event: FormEvent) {
    event.preventDefault()
    const normalized = query.trim()
    if (!normalized || !enabled) return
    setAnswering(true)
    setResult(null)
    setError(null)
    try {
      setResult(await answerQuestion(normalized))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The question could not be answered')
    } finally {
      setAnswering(false)
    }
  }

  function followCitation(contextId: string) {
    setFocusedContext(contextId)
    document.getElementById(`context-${contextId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  return (
    <Paper
      variant="outlined"
      sx={{
        flex: 1,
        width: '100%',
        minHeight: 520,
        p: { xs: 3, md: 5 },
        bgcolor: '#fff',
      }}
    >
      <Box component="form" onSubmit={submitQuestion}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
          <TextField
            fullWidth
            label="Ask a question"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            disabled={answering || !enabled}
            helperText={enabled ? 'The first answer may take longer while Qwen loads into GPU memory.' : 'Add a ready document before asking a question.'}
          />
          <Button
            type="submit"
            variant="contained"
            size="large"
            startIcon={answering ? <CircularProgress size={18} color="inherit" /> : <SearchRounded />}
            disabled={answering || !enabled || !query.trim()}
            sx={{ minWidth: 160, alignSelf: 'flex-start', height: 56 }}
          >
            {answering ? 'Grounding' : 'Query'}
          </Button>
        </Stack>
      </Box>

      {error && <Alert severity="error" sx={{ mt: 3 }}>{error}</Alert>}

      {answering && (
        <Stack alignItems="center" spacing={2} sx={{ py: 8 }} role="status">
          <CircularProgress />
          <Typography>Loading</Typography>
        </Stack>
      )}

      {result && (
        <Stack spacing={3} sx={{ mt: 4 }}>
          {result.status === 'insufficient_evidence' && (
            <Alert severity="warning">The retrieved passages did not provide enough evidence for an answer.</Alert>
          )}
          {result.status === 'generation_failed' && (
            <Alert severity="error">Generation failed its output or citation contract: {result.failure_reason ?? 'unknown reason'}.</Alert>
          )}
          {result.status === 'answered' && (
            <Paper variant="outlined" sx={{ p: { xs: 2.5, md: 3.5 }, borderLeft: '5px solid', borderLeftColor: 'primary.main' }}>
              <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 2 }}>
                <AutoAwesomeOutlined color="primary" />
                <Typography variant="h5" fontFamily="Georgia, serif">Grounded response</Typography>
              </Stack>
              <Stack spacing={2}>
                {result.claims.map((claim, index) => (
                  <Box key={`${claim.text}-${index}`}>
                    <Typography sx={{ lineHeight: 1.75 }}>{claim.text}</Typography>
                    <Stack direction="row" spacing={0.75} sx={{ mt: 0.75 }}>
                      {claim.citation_ids.map((citationId) => (
                        <Button key={citationId} size="small" variant="text" onClick={() => followCitation(citationId)}>
                          {citationId}
                        </Button>
                      ))}
                    </Stack>
                  </Box>
                ))}
              </Stack>
            </Paper>
          )}

          {!!result.contexts.length && (
            <Box>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
                <FindInPageOutlined color="primary" />
                <Typography variant="h5" fontFamily="Georgia, serif">Retrieved evidence</Typography>
              </Stack>
              <Stack spacing={1.5}>
                {result.contexts.map((context) => (
                  <Paper
                    id={`context-${context.context_id}`}
                    key={context.context_id}
                    variant="outlined"
                    sx={{
                      p: 2.5,
                      borderColor: focusedContext === context.context_id ? 'primary.main' : 'divider',
                      bgcolor: focusedContext === context.context_id ? 'rgba(23, 107, 98, 0.06)' : 'background.paper',
                      transition: 'background-color 160ms, border-color 160ms',
                    }}
                  >
                    <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={1.5}>
                      <Box>
                        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                          <Chip size="small" color="primary" label={context.context_id} />
                          <Typography fontWeight={700}>{context.document_name}</Typography>
                          <Typography variant="caption" color="text.secondary">{pageLabel(context)}</Typography>
                        </Stack>
                        <Typography sx={{ mt: 1.5, whiteSpace: 'pre-wrap', lineHeight: 1.65 }}>{context.text}</Typography>
                        <Stack direction="row" spacing={1} sx={{ mt: 1.5 }}>
                          {context.keyword_rank && <Chip size="small" variant="outlined" label={`Lexical #${context.keyword_rank}`} />}
                          {context.semantic_rank && <Chip size="small" variant="outlined" label={`Semantic #${context.semantic_rank}`} />}
                        </Stack>
                      </Box>
                      <Button
                        component="a"
                        href={sourceUrl(context)}
                        target="_blank"
                        rel="noreferrer"
                        size="small"
                        endIcon={<OpenInNewRounded />}
                        sx={{ alignSelf: 'flex-start', whiteSpace: 'nowrap' }}
                      >
                        Open source
                      </Button>
                    </Stack>
                  </Paper>
                ))}
              </Stack>
            </Box>
          )}
        </Stack>
      )}
    </Paper>
  )
}
