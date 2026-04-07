/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bolt: {
          50: '#eef7ff',
          100: '#d5e9ff',
          200: '#add4ff',
          300: '#7ab9ff',
          400: '#4298ff',
          500: '#1472ff',
          600: '#0a57d8',
          700: '#0846ad',
          800: '#0b3b8b',
          900: '#0f316f',
        },
        spark: {
          300: '#ffe872',
          400: '#ffd83d',
          500: '#f8c400',
        },
      },
      boxShadow: {
        neon: '0 0 18px rgba(66, 152, 255, 0.45), 0 0 40px rgba(248, 196, 0, 0.18)',
        pulse: '0 0 22px rgba(248, 216, 61, 0.5)',
      },
      backgroundImage: {
        grid: 'linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px)',
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        float: 'float 5s ease-in-out infinite',
        flicker: 'flicker 2.8s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        flicker: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.65' },
        },
      },
    },
  },
  plugins: [],
}
