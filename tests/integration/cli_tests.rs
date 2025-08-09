//! Integration tests for the CLI interface

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("ZKP Dataset Ledger"))
        .stdout(predicate::str::contains("init"))
        .stdout(predicate::str::contains("notarize"));
}

#[test]
fn test_cli_status() {
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.arg("status");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Ledger Status"))
        .stdout(predicate::str::contains("Total Entries"));
}

#[test]
fn test_cli_init_command() {
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.args(&["init", "--project", "test-project"]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Initializing ZKP Dataset Ledger"))
        .stdout(predicate::str::contains("Successfully initialized ledger"))
        .stdout(predicate::str::contains("test-project"));
}

#[test]
fn test_cli_notarize_command() {
    // Create a test CSV file
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "id,name,value").unwrap();
    writeln!(temp_file, "1,Alice,100").unwrap();
    writeln!(temp_file, "2,Bob,200").unwrap();

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.args(&[
        "notarize",
        temp_path.to_str().unwrap(),
        "--name",
        "test-dataset",
    ]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Notarizing dataset"))
        .stdout(predicate::str::contains("Successfully notarized"))
        .stdout(predicate::str::contains("test-dataset"));

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_cli_verify_chain() {
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.arg("verify-chain");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Verifying ledger chain integrity"))
        .stdout(predicate::str::contains("Chain integrity verification"));
}

#[test]
fn test_cli_transform_command() {
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.args(&[
        "transform",
        "--from",
        "dataset-a",
        "--to",
        "dataset-b",
        "--operation",
        "normalize",
        "--params",
        "method=minmax",
        "--params",
        "feature_range=0,1",
    ]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Recording transformation"))
        .stdout(predicate::str::contains("dataset-a -> dataset-b"));
}

#[test]
fn test_cli_split_command() {
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.args(&["split", "test-dataset", "--ratio", "0.8", "--seed", "42"]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Recording data split"))
        .stdout(predicate::str::contains("test-dataset"))
        .stdout(predicate::str::contains("Ratio: 0.80"));
}

#[test]
fn test_cli_history_command() {
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.args(&["history", "test-dataset", "--format", "summary"]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Dataset history"))
        .stdout(predicate::str::contains("test-dataset"));
}

#[test]
fn test_cli_export_command() {
    use tempfile::tempdir;
    let temp_dir = tempdir().unwrap();
    let export_path = temp_dir.path().join("export.json");

    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.args(&["export", export_path.to_str().unwrap(), "--format", "json"]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Exporting ledger data"))
        .stdout(predicate::str::contains("Exported to JSON"));
}
