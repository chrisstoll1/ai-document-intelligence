import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createTheme, CssBaseline, ThemeProvider } from '@mui/material'

import App from './App'

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#176b62' },
    secondary: { main: '#b65b3d' },
    background: { default: '#f2efe8', paper: '#fffdf8' },
    text: { primary: '#17262d', secondary: '#56666d' },
    divider: '#d8d4ca',
  },
  shape: { borderRadius: 4 },
  typography: {
    fontFamily: 'Inter, Segoe UI, sans-serif',
    button: { textTransform: 'none', fontWeight: 700 },
  },
  components: {
    MuiPaper: { styleOverrides: { root: { boxShadow: '0 14px 40px rgba(28, 44, 50, 0.06)' } } },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </StrictMode>,
)
