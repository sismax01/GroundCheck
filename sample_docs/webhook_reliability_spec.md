# Webhook Reliability & Idempotent Delivery Architecture (BridgeTest v2.4)

## 1. Cryptographic Signature Verification (HMAC-SHA256)
All outbound webhook events dispatched by the BridgeTest event broker are signed using `HMAC-SHA256`. The canonical signing string is constructed by concatenating the Unix epoch timestamp (in seconds), a literal ASCII period (`.`), and the raw UTF-8 JSON request body: `v1:{timestamp}.{raw_payload}`. Receivers must validate the `X-Bridge-Signature` header using a constant-time byte comparison (`hmac.compare_digest`) to prevent timing side-channel attacks.

To mitigate replay attacks, webhook consumers must enforce a replay tolerance window of $300\text{ seconds}$ ($5\text{ minutes}$) around the `X-Bridge-Timestamp` header. Any event with a timestamp older than $300\text{ seconds}$ or more than $30\text{ seconds}$ in the future must be rejected with HTTP `401 Unauthorized`.

## 2. Exponential Backoff with Full Jitter & Dead-Letter Queue (DLQ)
When a downstream webhook endpoint returns a transient error (`HTTP 408`, `429`, or `5xx`) or exceeds the per-attempt socket timeout of $8.0\text{ seconds}$, the dispatcher schedules retries using truncated exponential backoff with full jitter:
$$t_{\text{sleep}} = \text{Uniform}\left(0, \min\left(T_{\text{cap}}, T_{\text{base}} \cdot 2^{\text{attempt}}\right)\right)$$
where the base delay is $T_{\text{base}} = 2.0\text{ seconds}$ and the maximum backoff cap is $T_{\text{cap}} = 900\text{ seconds}$ ($15\text{ minutes}$). After $8$ consecutive failed delivery attempts, the event is marked `exhausted` and routed to the persistent Dead-Letter Queue (`dlq_webhook_events`) with a retention period of $14\text{ days}$.

## 3. Consumer-Side Idempotency and Deduplication Guarantees
Because network partitions can cause duplicate deliveries ("at-least-once" semantics), every webhook payload carries a globally unique UUIDv7 `X-Bridge-Delivery-Id` header. Consumers must execute state mutations inside an atomic transaction guarded by an idempotency key store with a minimum TTL of $24\text{ hours}$ ($86400\text{ seconds}$). If a duplicate `X-Bridge-Delivery-Id` arrives while an earlier request is still in-flight, the consumer returns `HTTP 409 Conflict` so the broker backs off safely.

## 4. Circuit Breaker State Machine and Plant Safety Interlock
To protect degraded downstream services, each target endpoint is wrapped in a three-state circuit breaker (`CLOSED`, `OPEN`, `HALF_OPEN`). The breaker trips from `CLOSED` to `OPEN` when the rolling failure rate exceeds $50\%$ across a minimum volume of $20$ requests within a $60\text{ second}$ window. Once `OPEN`, fast-fail mode blocks outbound calls for a cooldown period of $45\text{ seconds}$ before transitioning to `HALF_OPEN` and permitting $3$ probe requests.

Note on Industrial Gateway Integration (v2.4 Errata): For PLC/SCADA telemetry bridges connected to the QuadTank-FDI safety bus, the emergency pump shutdown timeout is configured at $120\text{ ms}$ to accommodate Modbus-TCP frame turnaround, superseding older controller defaults.
