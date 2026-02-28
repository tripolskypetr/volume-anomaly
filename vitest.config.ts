import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['test/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      include: ['src/**/*.ts'],
    },
  },
  resolve: {
    alias: {
      // позволяет в тестах писать: import { hawkesFit } from '#math'
      '#math': new URL('./src/math/index.ts', import.meta.url).pathname,
    },
  },
});
