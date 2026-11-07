# Signals

`models.py` defines immutable strategy signals. `validation.py` rejects invalid
prices, malformed direction/stop/target relationships, expired or future
timestamps, duplicate identities, and signals outside a configured market
session. Signal validation is separate from the risk engine, which remains the
authoritative pre-trade approval layer.
