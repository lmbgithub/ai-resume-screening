import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // standalone keeps the runtime image small: Next copies only the traced
  // server files, so the container does not ship node_modules.
  output: "standalone",
};

export default nextConfig;
