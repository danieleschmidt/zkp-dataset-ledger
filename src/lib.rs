pub mod circuits;
pub mod crypto;
pub mod dataset;
pub mod error;
pub mod ledger;
pub mod proof;
pub mod storage;

pub use dataset::Dataset;
pub use error::LedgerError;
pub use ledger::Ledger;
pub use proof::{Proof, ProofConfig};

pub type Result<T> = std::result::Result<T, LedgerError>;
