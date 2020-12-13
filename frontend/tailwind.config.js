module.exports = {
  purge: {
    enabled: true,
    content: ['../templates/**/*.jinja2']
  },
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {},
  },
  variants: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
