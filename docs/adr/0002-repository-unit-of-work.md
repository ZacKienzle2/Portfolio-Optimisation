# 2. Repository and Unit of Work

- Status: Accepted
- Date: 2026-06-20

## Context

The analytics and service layers need market data without knowing whether it
comes from a network vendor, a local cache, or a test fixture. Tests must run
without network access, and a future write side (positions, allocations) should
slot in without rewriting call sites.

## Decision

Market-data access is expressed as a `MarketDataRepository` protocol in the
domain layer. Concrete implementations live in the infrastructure layer:

- `YfinanceParquetRepository` fetches from the vendor and caches to parquet,
  validating the cached snapshot against the requested tickers and window before
  reuse.
- `FakeMarketDataRepository` serves in-memory frames for hermetic tests.

A `UnitOfWork` protocol bounds a run as a context manager. The in-memory
implementation provides commit and rollback semantics today and gives a future
write side a natural place to live without breaking existing call sites.

## Consequences

- The service layer depends on protocols, so the data source is swappable and
  the pipeline is testable offline.
- Cache correctness is centralised: a request that the snapshot does not cover
  triggers a refetch rather than silently serving stale data.
- The Unit of Work is intentionally minimal until a write side exists; the
  abstraction is in place but not over-built.
