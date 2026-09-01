import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Separate from vite.config.ts (which also loads @tailwindcss/vite) —
// component tests don't need Tailwind's build-time CSS processing, and
// keeping the two configs apart avoids type conflicts between Vite's
// and Vitest's augmented `defineConfig`.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
})
