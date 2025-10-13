import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'node', // Perfect for API bindings
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html']
    }
  }
})
