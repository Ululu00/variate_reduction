import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = REPO_ROOT / "scripts" / "run_mlp_variate_reduction_grid.py"
    spec = importlib.util.spec_from_file_location("run_mlp_variate_reduction_grid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunGridProtocolTest(unittest.TestCase):
    def test_baseline_only_dry_run_generates_no_candidate_runs(self):
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_mlp_variate_reduction_grid.py"),
            "--dry_run",
            "--gpu",
            "0",
            "--datasets",
            "Weather",
            "--pred_lens",
            "96",
            "--baseline_only",
            "--split_factor",
            "1",
            "--result_csv",
            "./results/test_validation_baseline_screens.csv",
            "--skip_test_eval",
            "--train_epochs",
            "1",
            "--patience",
            "1",
            "--num_workers",
            "0",
            "--no_wandb",
            "--wandb_mode",
            "disabled",
        ]

        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("[BASELINE]", result.stdout)
        self.assertIn("Weather pred_len=96 baseline seed=2021", result.stdout)
        self.assertIn("Total runs: 1", result.stdout)
        self.assertNotIn("method=", result.stdout)

    def test_completed_key_distinguishes_loss_temperature_and_budget(self):
        runner = load_runner()
        run = {
            "dataset": "Weather",
            "pred_len": 96,
            "k_spec": {
                "reduced_variate_k": 2,
                "target_k_ratio": 0.1,
                "target_k_value": 2,
                "k_selection_mode": "ratio",
                "k_ratio_denominator": "target",
            },
            "method": "mlp_generation",
            "variate_expansion_type": "transpose",
            "variate_decode_stage": "feature",
            "orthogonal_loss_weight": 0.01,
            "reconstruction_loss_weight": 0.0,
            "coverage_loss_weight": 0.0,
            "assignment_entropy_loss_weight": 0.0,
            "linear_weight_l2_loss_weight": 0.0,
            "decoder_init_l2_loss_weight": 0.0,
            "linear_coverage_loss_weight": 0.0,
            "biorthogonal_loss_weight": 0.0,
            "linear_cosine_loss_weight": 0.0,
            "mae_loss_weight": 0.0,
            "variate_token_split_factor": 1,
            "local_temporal_branch": "none",
            "local_temporal_init": "persistence",
            "local_temporal_gate_init": 1.0,
            "local_temporal_rank": 4,
            "zero_init_projector": False,
            "skip_backbone": False,
            "output_calibration": "none",
            "lowrank_reducer_rank": 8,
            "decoder_residual_gate_init": 0.0,
            "expansion_temperature": 1.0,
            "expansion_topk": 0,
            "backbone_residual_gate_init": 1.0,
            "backbone_residual_gate_type": "scalar",
            "variate_anchor_map_path": "",
            "selection_id": "",
            "selected_recipe": "",
            "manual_override": "0",
            "skip_test_eval": True,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "train_epochs": 10,
            "patience": 3,
            "num_workers": 0,
            "seed": 2021,
        }
        row = {
            "status": "success",
            "dataset": "Weather",
            "pred_len": "96",
            "reduced_variate_k": "2",
            "target_k_ratio": "0.1",
            "target_k_value": "2",
            "k_selection_mode": "ratio",
            "k_ratio_denominator": "target",
            "variate_reduction_type": "mlp_generation",
            "variate_expansion_type": "transpose",
            "variate_decode_stage": "feature",
            "orthogonal_loss_weight": "0.01",
            "reconstruction_loss_weight": "0.0",
            "coverage_loss_weight": "0.01",
            "assignment_entropy_loss_weight": "0.0",
            "linear_weight_l2_loss_weight": "0.0",
            "decoder_init_l2_loss_weight": "0.0",
            "linear_coverage_loss_weight": "0.0",
            "biorthogonal_loss_weight": "0.0",
            "linear_cosine_loss_weight": "0.0",
            "mae_loss_weight": "0.0",
            "variate_token_split_factor": "1",
            "local_temporal_branch": "none",
            "local_temporal_init": "persistence",
            "local_temporal_gate_init": "1.0",
            "local_temporal_rank": "4",
            "zero_init_projector": "False",
            "skip_backbone": "False",
            "output_calibration": "none",
            "lowrank_reducer_rank": "8",
            "decoder_residual_gate_init": "0.0",
            "expansion_temperature": "1.0",
            "expansion_topk": "0",
            "backbone_residual_gate_init": "1.0",
            "backbone_residual_gate_type": "scalar",
            "variate_anchor_map_path": "",
            "selection_id": "",
            "selected_recipe": "",
            "manual_override": "0",
            "eval_split": "val",
            "batch_size": "32",
            "learning_rate": "0.0001",
            "train_epochs": "10",
            "patience": "3",
            "num_workers": "0",
            "seed": "2021",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "done.csv"
            fieldnames = list(row)
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self.assertNotIn(runner.run_key(run), runner.completed_keys_from_csv(csv_path))

            row["coverage_loss_weight"] = "0.0"
            row["expansion_temperature"] = "0.5"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self.assertNotIn(runner.run_key(run), runner.completed_keys_from_csv(csv_path))

            row["expansion_temperature"] = "1.0"
            row["train_epochs"] = "1"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self.assertNotIn(runner.run_key(run), runner.completed_keys_from_csv(csv_path))

            row["train_epochs"] = "10"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self.assertIn(runner.run_key(run), runner.completed_keys_from_csv(csv_path))

            run["linear_cosine_loss_weight"] = 0.001
            self.assertNotIn(runner.run_key(run), runner.completed_keys_from_csv(csv_path))

            row["linear_cosine_loss_weight"] = "0.001"
            fieldnames = list(row)
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self.assertIn(runner.run_key(run), runner.completed_keys_from_csv(csv_path))

            run["k_spec"]["k_ratio_denominator"] = "source"
            self.assertNotIn(runner.run_key(run), runner.completed_keys_from_csv(csv_path))

            row["k_ratio_denominator"] = "source"
            fieldnames = list(row)
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self.assertIn(runner.run_key(run), runner.completed_keys_from_csv(csv_path))

    def test_fixed_k_exceeding_original_v_is_skipped_with_split_tokens(self):
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_mlp_variate_reduction_grid.py"),
            "--dry_run",
            "--datasets",
            "Weather",
            "--pred_lens",
            "96",
            "--methods",
            "mlp_generation",
            "--fixed_k_values",
            "32",
            "--no_ratio_k",
            "--no_enforce_k_budget",
            "--split_factor",
            "3",
            "--orthogonal_loss_weight",
            "0",
            "--reconstruction_loss_weight",
            "0",
            "--linear_weight_l2_loss_weight",
            "0",
            "--seeds",
            "2021",
            "--no_baseline",
            "--no_wandb",
            "--wandb_mode",
            "disabled",
        ]

        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("K=32 exceeds original variables V=21", result.stdout)
        self.assertIn("SKIP=", result.stdout)

    def test_ratio_denominator_both_emits_target_and_source_k(self):
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_mlp_variate_reduction_grid.py"),
            "--dry_run",
            "--datasets",
            "Weather",
            "--pred_lens",
            "96",
            "--methods",
            "mlp_generation",
            "--k_ratios",
            "0.1",
            "--k_ratio_denominator",
            "both",
            "--no_enforce_k_budget",
            "--split_factor",
            "3",
            "--orthogonal_loss_weight",
            "0",
            "--reconstruction_loss_weight",
            "0",
            "--linear_weight_l2_loss_weight",
            "0",
            "--seeds",
            "2021",
            "--no_baseline",
            "--no_wandb",
            "--wandb_mode",
            "disabled",
        ]

        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("ratio_basis=target ratio=0.10 K=2", result.stdout)
        self.assertIn("ratio_basis=source ratio=0.10 K=6", result.stdout)

    def test_no_fixed_k_and_loss_variant_names_make_ratio_only_named_run(self):
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_mlp_variate_reduction_grid.py"),
            "--dry_run",
            "--sweep_preset",
            "mlp_generation_loss_goal",
            "--datasets",
            "Weather",
            "--pred_lens",
            "96",
            "--k_ratios",
            "0.1",
            "--k_ratio_denominator",
            "target",
            "--no_fixed_k",
            "--loss_variant_names",
            "mse_only",
            "--seeds",
            "2021",
            "--no_baseline",
            "--no_wandb",
            "--wandb_mode",
            "disabled",
        ]

        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("variant=mse_only", result.stdout)
        self.assertIn("ratio_basis=target ratio=0.10 K=2", result.stdout)
        self.assertIn("Total runs: 1", result.stdout)
        self.assertNotIn(" split=1 K=4 actual_ratio", result.stdout)


if __name__ == "__main__":
    unittest.main()
