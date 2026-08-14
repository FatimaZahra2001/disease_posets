"""Thin, diagnostics-preserving interface to the vendored official PyMBA."""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np


def import_true_pymba():
    """Import the official vendored modules directly from their source tree."""
    source = Path(__file__).resolve().parents[1] / "external" / "gkazunii_pymba" / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Official PyMBA source not found: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import manage_intract
    import mask
    import mproject
    import transform
    return {
        "manage_intract": manage_intract,
        "mask": mask,
        "mproject": mproject,
        "transform": transform,
        "module_path": source_text,
    }


def body_indices(body, backend=None):
    backend = backend or import_true_pymba()
    interactions = backend["manage_intract"].get_m_body_intract(body, 3)
    return backend["mask"].get_learn_indices((3, 3, 3), interactions)


def kl_pq(P, Q):
    """KL(P||Q), avoiding overflow in an intermediate P/Q ratio."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if np.any(P <= 0) or np.any(Q <= 0):
        raise FloatingPointError("KL requires strictly positive probability tensors")
    return float(np.sum(P * (np.log(P) - np.log(Q))))


def official_project(P, body, max_iter=1000, tol=1e-8, init="uniform", method="lbfgs", seed=42):
    """Run the validated wrapper around the official MBA implementation."""
    backend = import_true_pymba()
    mproject = backend["mproject"]
    transform = backend["transform"]
    P = np.asarray(P, float)
    P = P / P.sum()
    captured = {}
    original_minimize = mproject.minimize

    def spying_minimize(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        captured["result"] = result
        return result

    stdout = io.StringIO()
    start = time.perf_counter()
    theta0 = transform.theta_from_prob(P, chi=1) if body == 3 else None
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
        if method == "lbfgs":
            mproject.minimize = spying_minimize
            try:
                Q, theta, eta, history = mproject.MBA_LBFGS(
                    P, body, max_iter=max_iter, seed=seed, init=init,
                    theta_0=theta0, verbose=False, tol=tol, gtol=tol,
                    get_history=True, get_cost_dual=True,
                    get_cost_alpha=False, chi=1,
                )
            finally:
                mproject.minimize = original_minimize
            result = captured.get("result")
        else:
            Q, theta, eta, history = mproject.MBA(
                P, body, max_iter=max_iter, seed=seed, init=init,
                theta_0=theta0, verbose=False, tol=tol, get_history=True,
                get_cost_dual=True, get_cost_alpha=False, chi=1,
            )
            result = None
    elapsed = time.perf_counter() - start
    Q = np.asarray(Q, float)
    Q /= Q.sum()
    indices = body_indices(body, backend)
    eta_target = transform.eta_from_prob(P, chi=1)
    eta_q = transform.eta_from_prob(Q, chi=1)
    mismatch = eta_q[tuple(indices.T)] - eta_target[tuple(indices.T)]
    theta_official = np.asarray(theta, float)
    reconstructed_theta = transform.theta_from_prob(Q, chi=1)
    forbidden = [abs(theta_official[index]) for index in np.ndindex(theta_official.shape)
                 if sum(value > 0 for value in index) > body]
    reconstructed_forbidden = [abs(reconstructed_theta[index])
                               for index in np.ndindex(reconstructed_theta.shape)
                               if sum(value > 0 for value in index) > body]
    grad_norm = float(np.linalg.norm(mismatch))
    success = bool(result.success) if result is not None else grad_norm <= tol
    diagnostic = {
        "body_order": body,
        "optimizer_success": success,
        "status_code": int(result.status) if result is not None else 0,
        "termination_message": str(result.message) if result is not None else "gradient convergence policy",
        "n_iterations": int(result.nit) if result is not None else len(history["iter"]),
        "final_loss": kl_pq(P, Q),
        "gradient_norm": float(np.linalg.norm(result.jac)) if result is not None else grad_norm,
        "active_eta_mismatch_max": float(np.max(np.abs(mismatch), initial=0)),
        "forbidden_theta_max": float(max(forbidden or [0.0])),
        "reconstructed_forbidden_theta_max": float(max(reconstructed_forbidden or [0.0])),
        "elapsed_seconds": elapsed,
        "kl_p_q": kl_pq(P, Q),
        "p_q_l2": float(np.linalg.norm(P - Q)),
        "q_sum": float(Q.sum()),
        "q_min": float(Q.min()),
        "q_max": float(Q.max()),
        "nonfinite": int((~np.isfinite(Q)).sum()),
    }
    if not np.isfinite(Q).all() or np.any(Q < 0) or not np.isclose(Q.sum(), 1, atol=1e-10):
        raise FloatingPointError("Invalid MBA Q")
    if diagnostic["forbidden_theta_max"] > max(1e-7, 10 * tol):
        raise AssertionError(
            "Forbidden theta retained: "
            f"body={body} max_abs={diagnostic['forbidden_theta_max']:.17g} "
            f"threshold={max(1e-7, 10 * tol):.17g} q_min={diagnostic['q_min']:.17g}"
        )
    if body == 3 and not np.allclose(Q, P, atol=max(1e-8, 10 * tol)):
        raise AssertionError("Body-3 identity failed")
    return Q, theta, eta, history, diagnostic


def backend_info():
    backend = import_true_pymba()

    return {
        "module_path": backend["module_path"],
        "official_call": "mproject.MBA_LBFGS(P, body)",
    }


def get_transform():
    """Return the transform module from the official backend."""
    return import_true_pymba()["transform"]


def project(
    P,
    body,
    *,
    max_iter,
    tol,
    init,
    seed,
):
    """Perform one official MBA projection.

    No post-hoc modification of theta, eta, or optimisation history occurs.
    """

    return official_project(
        P,
        body,
        max_iter=max_iter,
        tol=tol,
        init=init,
        method="lbfgs",
        seed=seed,
    )


__all__ = [
    "backend_info",
    "body_indices",
    "get_transform",
    "import_true_pymba",
    "kl_pq",
    "official_project",
    "project",
]
