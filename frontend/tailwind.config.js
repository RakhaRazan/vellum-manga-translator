/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        studio: {
          canvas: 'rgb(var(--studio-canvas) / <alpha-value>)',
          panel: 'rgb(var(--studio-panel) / <alpha-value>)',
          surface: 'rgb(var(--studio-surface) / <alpha-value>)',
          hover: 'rgb(var(--studio-hover) / <alpha-value>)',
          border: 'rgb(var(--studio-border) / <alpha-value>)',
          'border-strong': 'rgb(var(--studio-border-strong) / <alpha-value>)',
          text: 'rgb(var(--studio-text) / <alpha-value>)',
          'text-secondary': 'rgb(var(--studio-text-secondary) / <alpha-value>)',
          'text-muted': 'rgb(var(--studio-text-muted) / <alpha-value>)',
          accent: 'rgb(var(--studio-accent) / <alpha-value>)',
          'accent-hover': 'rgb(var(--studio-accent-hover) / <alpha-value>)',
          'accent-active': 'rgb(var(--studio-accent-active) / <alpha-value>)',
          'accent-fg': 'rgb(var(--studio-accent-fg) / <alpha-value>)',
          success: 'rgb(var(--studio-success) / <alpha-value>)',
          danger: 'rgb(var(--studio-danger) / <alpha-value>)',
          warning: 'rgb(var(--studio-warning) / <alpha-value>)',
          info: 'rgb(var(--studio-info) / <alpha-value>)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        comic: ['"Comic Neue"', '"Anime Ace"', '"Comic Sans MS"', 'cursive', 'sans-serif'],
      },
      borderRadius: {
        'studio-sm': '2px',
        studio: '4px',
        'studio-md': '6px',
      },
    },
  },
  plugins: [],
};
