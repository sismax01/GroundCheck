# Raft Distributed Consensus & Write-Ahead Log Storage Specification

## 1. Leader Election Timers and Split-Vote Prevention
The distributed state machine maintains fault-tolerant consensus across $N = 2f + 1$ replicas (tolerating up to $f$ crash-stop node failures; $f = 2$ for a $5$-node cluster). Followers monitor leader liveness using a randomized election timeout drawn uniformly from the interval $[150\text{ ms}, 300\text{ ms}]$, while the active leader broadcasts `AppendEntries` heartbeats every $50\text{ ms}$. Randomizing the election timeout prevents deterministic split votes during simultaneous candidate transitions.

## 2. Write-Ahead Log (WAL) Segment Layout and CRC32C Integrity
Every state machine mutation is appended to a pre-allocated $64\text{ MB}$ Write-Ahead Log (WAL) segment file before acknowledgment. Each WAL frame consists of a $16\text{ byte}$ binary header containing a $4\text{ byte}$ Castagnoli `CRC32C` checksum, a $4\text{ byte}$ payload length, and an $8\text{ byte}$ monotonically increasing log index (`uint64`). Group commit batches concurrent client writes over a maximum coalescing window of $2.5\text{ ms}$ before invoking `fdatasync()` on the underlying NVMe block device.

## 3. Log Compaction, Snapshotting, and Follower Catch-Up
To bound disk usage and accelerate node recovery, the storage engine triggers background copy-on-write snapshot compaction whenever the active WAL exceeds $50000$ committed entries or $128\text{ MB}$ of uncompacted log data. If a lagging follower requests log entries that have already been garbage-collected prior to the latest snapshot index, the leader streams the compressed snapshot in $4\text{ MB}$ chunks via the `InstallSnapshot` RPC.

## 4. Linearizable Read Leases and Quorum Verification
To serve linearizable reads (`ReadIndex`) without appending a no-op entry to the Raft log, the leader records its current `commitIndex`, confirms quorum leadership via a lightweight heartbeat round to a majority ($\lceil (N+1)/2 \rceil$) of peers, and waits until its local state machine `lastApplied` index reaches or exceeds `commitIndex`. Clock-drift-bounded lease reads are disabled by default in multi-region deployments where NTP clock skew can exceed $15\text{ ms}$.
