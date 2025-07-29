pub mod circuits;
pub mod crypto;
pub mod ledger;
pub mod storage;
pub mod proof;
pub mod dataset;
pub mod error;

pub use ledger::Ledger;
pub use dataset::Dataset;
pub use proof::{Proof, ProofConfig};
pub use error::LedgerError;

pub type Result<T> = std::result::Result<T, LedgerError>;