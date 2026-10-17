import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import globals from 'globals'
import prettier from 'eslint-config-prettier'

export default [
  // src/generated is written by `dsbf schema export`.
  { ignores: ['dist/', 'node_modules/', 'src/generated/'] },
  js.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  {
    files: ['src/**/*.{js,vue}'],
    languageOptions: { globals: globals.browser },
  },
  {
    files: ['*.js'],
    languageOptions: { globals: globals.node },
  },
  // Existing violations the toolchain unit didn't refactor away. They stay visible as
  // warnings; each is listed as a follow-up in the roadmap.
  {
    files: ['src/**/*.{js,vue}'],
    rules: {
      'no-unused-vars': 'warn',
      'vue/no-side-effects-in-computed-properties': 'warn',
      // highlightHtml() escapes the text before it is rendered with v-html.
      'vue/no-v-html': 'off',
      'vue/require-default-prop': 'warn',
    },
  },
  // Last, so it switches off every formatting rule that Prettier owns.
  prettier,
]
