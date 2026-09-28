# ==============================================================================
# Kavach-Opt Production Multi-Stage Dockerfile (Render / Linux Deployment)
# ==============================================================================
# Stage 1: Build C++20 Optimization Solver Core & React / Vite Frontend
# ==============================================================================
FROM node:22-bookworm-slim AS builder

# Install C++20 build toolchain (GCC 12, CMake 3.25+, Make)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Compile C++20 Solver Core in Release mode
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/

RUN cmake -B build-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DKAVACH_ENABLE_WARNINGS_AS_ERRORS=OFF \
    -DKAVACH_BUILD_TESTS=OFF \
    -DKAVACH_BUILD_CLI=ON \
 && cmake --build build-release --target kavach-opt -j$(nproc)

# 2. Build Frontend UI Bundle
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Minimal Production Runtime
# ==============================================================================
FROM node:22-bookworm-slim AS runner

WORKDIR /app

# Copy native C++ executable from builder
COPY --from=builder /app/build-release/kavach-opt /app/build-release/kavach-opt
RUN chmod +x /app/build-release/kavach-opt

# Copy built frontend assets and Node API server
COPY --from=builder /app/frontend/dist /app/frontend/dist
COPY frontend/server /app/frontend/server
COPY frontend/package.json /app/frontend/package.json

# Copy benchmark models, refinery examples, and verified baseline artifacts
COPY benchmarks /app/benchmarks
COPY examples /app/examples
COPY benchmark_results /app/benchmark_results

# Production configuration
ENV NODE_ENV=production
ENV PORT=10000
ENV HOST=0.0.0.0
ENV KAVACH_BIN_PATH=/app/build-release/kavach-opt

EXPOSE 10000

# Start unified Node API + Static Frontend Server
CMD ["node", "frontend/server/index.mjs"]
