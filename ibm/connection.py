"""
ibm/connection.py
=================
Open an IBM Quantum ``QiskitRuntimeService`` from credentials in a ``.env`` file.

The ``.env`` lives at the repository root (``codes/.env`` or the project root)
and is git-ignored — it holds the private token and must never be committed.
Expected keys (only the token is strictly required; instance/channel have
sensible defaults for the current IBM Quantum Platform)::

    IBM_QUANTUM_TOKEN=<your API token>
    IBM_QUANTUM_INSTANCE=<CRN or hub/group/project>     # optional
    IBM_QUANTUM_CHANNEL=ibm_quantum_platform            # optional, default shown

The loader also accepts the common alternate names ``QISKIT_IBM_TOKEN`` /
``QISKIT_IBM_INSTANCE`` / ``QISKIT_IBM_CHANNEL`` and, failing a ``.env`` file,
the process environment — so a session credential works without a file.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---- credential loading -----------------------------------------------------

_TOKEN_KEYS = ("IBM_QUANTUM_TOKEN", "QISKIT_IBM_TOKEN")
_INSTANCE_KEYS = ("IBM_QUANTUM_INSTANCE", "QISKIT_IBM_INSTANCE")
_CHANNEL_KEYS = ("IBM_QUANTUM_CHANNEL", "QISKIT_IBM_CHANNEL")
_DEFAULT_CHANNEL = "ibm_quantum_platform"


def find_dotenv(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: this file's dir) looking for a .env."""
    here = (start or Path(__file__).resolve().parent)
    for d in (here, *here.parents):
        cand = d / ".env"
        if cand.is_file():
            return cand
    return None


def load_credentials(dotenv_path: str | os.PathLike | None = None) -> dict:
    """
    Return ``{"token","instance","channel"}`` from .env (preferred) or the
    process environment. Raises ``RuntimeError`` if no token is found.

    Values already present in ``os.environ`` win over the .env file, so a
    session-injected credential overrides a stale file.
    """
    # 1) load .env into a local dict WITHOUT clobbering real env vars
    file_vals: dict[str, str] = {}
    path = Path(dotenv_path) if dotenv_path else find_dotenv()
    if path and path.is_file():
        try:
            from dotenv import dotenv_values
            file_vals = {k: v for k, v in dotenv_values(path).items() if v is not None}
        except ImportError:
            # minimal fallback parser (KEY=VALUE, ignore # comments / blanks)
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                file_vals[k.strip()] = v.strip().strip('"').strip("'")

    def pick(keys, default=None):
        for k in keys:
            if os.environ.get(k):          # process env wins
                return os.environ[k]
        for k in keys:
            if file_vals.get(k):
                return file_vals[k]
        return default

    token = pick(_TOKEN_KEYS)
    if not token:
        raise RuntimeError(
            "No IBM Quantum token found. Set IBM_QUANTUM_TOKEN in codes/.env "
            "(git-ignored) or in the session environment. Never commit the token."
        )
    return {
        "token": token,
        "instance": pick(_INSTANCE_KEYS),
        "channel": pick(_CHANNEL_KEYS, _DEFAULT_CHANNEL),
    }


# ---- service handle ---------------------------------------------------------

def get_service(dotenv_path: str | os.PathLike | None = None):
    """
    Build a ``QiskitRuntimeService`` from the loaded credentials.

    Import of ``qiskit_ibm_runtime`` is deferred so this module imports on a
    machine without the runtime installed.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    creds = load_credentials(dotenv_path)
    kwargs = {"channel": creds["channel"], "token": creds["token"]}
    if creds["instance"]:
        kwargs["instance"] = creds["instance"]
    return QiskitRuntimeService(**kwargs)


def pick_backend(service, min_qubits: int = 5, name: str | None = None,
                 prefer_least_busy: bool = True):
    """
    Resolve a backend: an explicit *name* if given, else the least-busy
    operational real device with at least *min_qubits* qubits.
    """
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False,
                              min_num_qubits=min_qubits)
