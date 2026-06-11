# Agent Runtime Configuration Guide

## Overview

This synthetic technical document describes a safe staging configuration for the Agent OS RAG pipeline.

## API Endpoint

Use the API endpoint `/api/v1/rag/search` for retrieval checks and `/api/v1/rag/ask` for extractive question answering.

## Configuration Keys

The primary configuration key is `RAG_HYBRID_BACKEND`.
The native Qdrant collection key is `QDRANT_HYBRID_COLLECTION`.
The sparse vector key is `QDRANT_SPARSE_VECTOR_NAME`.

## Function Names

The validation helper function is `validate_hybrid_backend()`.
The fallback helper function is `fallback_to_python_bm25()`.

## Risks

The main risk is stale sparse vectors after schema changes.
Another risk is incomplete reingestion, which can make Qdrant hybrid results look weaker than Python BM25.
Always compare staging results before changing the default backend.
