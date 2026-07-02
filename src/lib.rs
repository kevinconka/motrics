//! `motrics` — an extremely fast MOT and HOTA metrics library.
//!
//! This crate is compiled into the `motrics._motrics` extension module. The
//! public, ergonomic API lives in the `motrics` Python package (see
//! `python/motrics/`), which re-exports the pieces below.
//!
//! At this stage the module only exposes a version placeholder that proves the
//! full Rust -> PyO3 -> Python build pipeline works end to end. Metric
//! implementations land in follow-up PRs.

use pyo3::prelude::*;

/// Return the version of the compiled Rust core.
///
/// Wired through Cargo at build time so the native and Python versions can
/// never drift.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The `motrics._motrics` extension module.
#[pymodule]
fn _motrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
