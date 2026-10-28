import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import globals from 'globals'
import prettier from 'eslint-config-prettier'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  // src/generated is written by `dsbf schema export`.
  { ignores: ['dist/', 'node_modules/', 'src/generated/'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  {
    files: ['src/**/*.{ts,vue}'],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { parser: tseslint.parser },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': 'error',
      'vue/no-side-effects-in-computed-properties': 'error',
      'vue/require-default-prop': 'error',
    },
  },
  {
    files: ['*.{js,ts}'],
    languageOptions: { globals: globals.node },
  },
  // Last, so it switches off every formatting rule that Prettier owns.
  prettier,
)
