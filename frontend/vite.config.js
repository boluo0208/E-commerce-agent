import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const BACKEND = process.env.VITE_BACKEND || 'http://127.0.0.1:9000'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: parseInt(process.env.VITE_PORT) || 5173,
    proxy: {
      '/api': BACKEND,
      '/health': BACKEND,
    },
  },
})
