use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::test]
async fn test_cli_ledger_initialization() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let project_name = format!("test-project-{}", Uuid::new_v4());

    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.current_dir(&temp_dir)
        .arg("init")
        .arg("--project")
        .arg(&project_name);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Ledger initialized"));
}

#[tokio::test]
async fn test_cli_dataset_notarization() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let project_name = format!("test-notarize-{}", Uuid::new_v4());

    // Initialize ledger
    let mut init_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    init_cmd
        .current_dir(&temp_dir)
        .arg("init")
        .arg("--project")
        .arg(&project_name);

    init_cmd.assert().success();

    // Create test CSV
    let csv_content = "name,age,city\nAlice,25,NYC\nBob,30,LA\nCharlie,35,Chicago";
    let csv_path = temp_dir.path().join("test_data.csv");
    std::fs::write(&csv_path, csv_content).expect("Failed to write test CSV");

    // Notarize dataset
    let mut notarize_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    notarize_cmd
        .current_dir(&temp_dir)
        .arg("notarize")
        .arg(csv_path.to_str().unwrap())
        .arg("--name")
        .arg("test-dataset-v1");

    notarize_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("Dataset notarized"));
}

#[tokio::test]
async fn test_cli_transformation_recording() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let project_name = format!("test-transform-{}", Uuid::new_v4());

    // Initialize and setup dataset
    let mut init_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    init_cmd
        .current_dir(&temp_dir)
        .arg("init")
        .arg("--project")
        .arg(&project_name);

    init_cmd.assert().success();

    let csv_content = "name,age,salary\nAlice,25,50000\nBob,30,60000\nCharlie,35,70000";
    let csv_path = temp_dir.path().join("original.csv");
    std::fs::write(&csv_path, csv_content).expect("Failed to write CSV");

    // Notarize original
    let mut notarize_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    notarize_cmd
        .current_dir(&temp_dir)
        .arg("notarize")
        .arg(csv_path.to_str().unwrap())
        .arg("--name")
        .arg("original-data");

    notarize_cmd.assert().success();

    // Record transformation
    let mut transform_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    transform_cmd
        .current_dir(&temp_dir)
        .arg("transform")
        .arg("--input")
        .arg("original-data")
        .arg("--output")
        .arg("normalized-data")
        .arg("--operation")
        .arg("normalize");

    transform_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("Transformation recorded"));
}

#[tokio::test]
async fn test_cli_audit_generation() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let project_name = format!("test-audit-{}", Uuid::new_v4());

    // Setup ledger with data
    let mut init_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    init_cmd
        .current_dir(&temp_dir)
        .arg("init")
        .arg("--project")
        .arg(&project_name);

    init_cmd.assert().success();

    let csv_content = "feature1,feature2,label\n1.0,2.0,A\n2.0,3.0,B\n3.0,4.0,A";
    let csv_path = temp_dir.path().join("training.csv");
    std::fs::write(&csv_path, csv_content).expect("Failed to write CSV");

    let mut notarize_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    notarize_cmd
        .current_dir(&temp_dir)
        .arg("notarize")
        .arg(csv_path.to_str().unwrap())
        .arg("--name")
        .arg("training-set");

    notarize_cmd.assert().success();

    // Generate audit report
    let audit_output = temp_dir.path().join("audit-report.json");
    let mut audit_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    audit_cmd
        .current_dir(&temp_dir)
        .arg("audit")
        .arg("--from")
        .arg("genesis")
        .arg("--to")
        .arg("latest")
        .arg("--format")
        .arg("json")
        .arg("--output")
        .arg(audit_output.to_str().unwrap());

    audit_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("Audit report generated"));

    assert!(audit_output.exists(), "Audit report file should exist");
}

#[tokio::test]
async fn test_cli_proof_verification() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let project_name = format!("test-verify-{}", Uuid::new_v4());

    // Setup and notarize data
    let mut init_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    init_cmd
        .current_dir(&temp_dir)
        .arg("init")
        .arg("--project")
        .arg(&project_name);

    init_cmd.assert().success();

    let csv_content = "x,y\n1,2\n3,4\n5,6";
    let csv_path = temp_dir.path().join("data.csv");
    std::fs::write(&csv_path, csv_content).expect("Failed to write CSV");

    let mut notarize_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    notarize_cmd
        .current_dir(&temp_dir)
        .arg("notarize")
        .arg(csv_path.to_str().unwrap())
        .arg("--name")
        .arg("verification-test");

    notarize_cmd.assert().success();

    // Verify proof
    let mut verify_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    verify_cmd
        .current_dir(&temp_dir)
        .arg("verify")
        .arg("--dataset")
        .arg("verification-test");

    verify_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("Proof verification"));
}

#[tokio::test]
async fn test_cli_dataset_splitting() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let project_name = format!("test-split-{}", Uuid::new_v4());

    // Setup ledger and data
    let mut init_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    init_cmd
        .current_dir(&temp_dir)
        .arg("init")
        .arg("--project")
        .arg(&project_name);

    init_cmd.assert().success();

    let csv_content = "feature,label\n1.0,A\n2.0,B\n3.0,A\n4.0,B\n5.0,A\n6.0,B\n7.0,A\n8.0,B";
    let csv_path = temp_dir.path().join("full_dataset.csv");
    std::fs::write(&csv_path, csv_content).expect("Failed to write CSV");

    let mut notarize_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    notarize_cmd
        .current_dir(&temp_dir)
        .arg("notarize")
        .arg(csv_path.to_str().unwrap())
        .arg("--name")
        .arg("full-dataset");

    notarize_cmd.assert().success();

    // Split dataset
    let mut split_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    split_cmd
        .current_dir(&temp_dir)
        .arg("split")
        .arg("--input")
        .arg("full-dataset")
        .arg("--train-ratio")
        .arg("0.7")
        .arg("--seed")
        .arg("42");

    split_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("Dataset split completed"));
}

#[tokio::test]
async fn test_cli_help_commands() {
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.arg("--help");

    cmd.assert().success().stdout(predicate::str::contains(
        "Zero-Knowledge Proof Dataset Ledger",
    ));

    let mut subcmd = Command::cargo_bin("zkp-ledger").unwrap();
    subcmd.args(&["notarize", "--help"]);

    subcmd
        .assert()
        .success()
        .stdout(predicate::str::contains("Notarize a dataset"));
}

#[tokio::test]
async fn test_cli_error_handling() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Test nonexistent file
    let mut cmd = Command::cargo_bin("zkp-ledger").unwrap();
    cmd.current_dir(&temp_dir)
        .arg("notarize")
        .arg("nonexistent.csv")
        .arg("--name")
        .arg("test");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Error"));

    // Test invalid command
    let mut invalid_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    invalid_cmd.arg("invalid-command");

    invalid_cmd
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[tokio::test]
async fn test_cli_config_operations() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let project_name = format!("test-config-{}", Uuid::new_v4());

    let mut init_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    init_cmd
        .current_dir(&temp_dir)
        .arg("init")
        .arg("--project")
        .arg(&project_name);

    init_cmd.assert().success();

    // Test config show
    let mut show_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    show_cmd.current_dir(&temp_dir).arg("config").arg("show");

    show_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("Configuration"));

    // Test config set
    let mut set_cmd = Command::cargo_bin("zkp-ledger").unwrap();
    set_cmd
        .current_dir(&temp_dir)
        .arg("config")
        .arg("set")
        .arg("storage.backend")
        .arg("rocksdb");

    set_cmd.assert().success();
}
