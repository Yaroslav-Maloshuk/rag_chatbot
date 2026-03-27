import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ink: '#102A43',
        sky: '#E8F5FF',
        mint: '#D6F7E2',
        sand: '#FFF6E8',
        coral: '#FF7F50',
        slate: '#2F3A4A',
      },
      fontFamily: {
        display: ['"Space Grotesk"', '"Manrope"', 'sans-serif'],
      },
      boxShadow: {
        soft: '0 14px 34px rgba(16, 42, 67, 0.10)',
      },
    },
  },
  plugins: [],
} satisfies Config;
