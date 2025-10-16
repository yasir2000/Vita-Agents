/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*', // Proxy to Backend
      },
    ]
  },
  env: {
    VITA_AGENTS_API_URL: process.env.VITA_AGENTS_API_URL || 'http://localhost:8000',
  },
}

module.exports = nextConfig