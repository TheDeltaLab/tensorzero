# Global build args
ARG CARGO_BUILD_FLAGS=""
ARG PROFILE="performance"

# ========== builder ==========

FROM lukemathwalker/cargo-chef:latest-rust-1.94-trixie AS chef

WORKDIR /src

FROM chef AS planner
WORKDIR /app/crates
COPY crates/ .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
ARG CARGO_BUILD_FLAGS
ARG PROFILE
WORKDIR /tensorzero
ENV CARGO_INCREMENTAL=0
RUN apt-get update && apt-get install -y clang libc++-dev && rm -rf /var/lib/apt/lists/*
COPY --from=planner /app/crates/recipe.json /tensorzero/crates/recipe.json
COPY crates/rust-toolchain.toml /tensorzero/crates/rust-toolchain.toml
WORKDIR /tensorzero/crates
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/tensorzero/crates/target,sharing=locked \
    cargo chef cook --profile ${PROFILE} -p gateway ${CARGO_BUILD_FLAGS}
COPY crates/ /tensorzero/crates/
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/tensorzero/crates/target,sharing=locked \
    cargo build --profile ${PROFILE} -p gateway ${CARGO_BUILD_FLAGS} && \
    if [ "${PROFILE}" = "dev" ]; then cp -r target/debug /release; else cp -r target/${PROFILE} /release; fi

# ========== base ==========

FROM debian:trixie-slim AS base

RUN apt-get update && apt-get install -y ca-certificates openssl wget && rm -rf /var/lib/apt/lists/*

# ========== gateway ==========

FROM base AS gateway

RUN useradd -m -s /bin/bash gateway

USER gateway

COPY --from=builder /release/gateway /usr/local/bin/gateway

WORKDIR /app

ENV TENSORZERO_GATEWAY_BIND_ADDRESS=0.0.0.0:3720

EXPOSE 3720

ENTRYPOINT ["gateway"]
