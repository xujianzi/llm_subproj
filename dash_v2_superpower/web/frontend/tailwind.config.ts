import type { Config } from 'tailwindcss'

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#0f1117',
        panel:   '#1a1d27',
        border:  '#2a2d3e',
        primary: '#4f7cff',
        accent:  '#00d4aa',
        text:    '#e2e8f0',
        muted:   '#94a3b8',
      },
    },
  },
  plugins: [],
} satisfies Config
