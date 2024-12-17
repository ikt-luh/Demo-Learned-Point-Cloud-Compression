import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    host: '0.0.0.0',  // Make it accessible from any network interface (important for Docker)
    port: 5173,        // The desired port for the frontend (you can change it)
    strictPort: true,  // If the port is already in use, Vite will throw an error rather than trying another port
  }
})