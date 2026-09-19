import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.json").open("r", encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_shared_protocol(self):
        defaults = self.config["defaults"]
        self.assertEqual(defaults["seeds"], [2024, 2025, 2026, 2027, 2028])
        self.assertEqual(defaults["dim"], 64)
        self.assertEqual(defaults["train_batch"], 512)
        self.assertEqual(defaults["epochs"], 300)
        self.assertEqual(defaults["p"], 5)
        self.assertGreater(defaults["temperature"], 0.0)
        self.assertNotIn("mc_samples", defaults)

    def test_dataset_specific_presets_are_not_published(self):
        self.assertEqual(set(self.config), {"defaults"})

    def test_readme_math_is_github_compatible(self):
        raw = (ROOT / "README.md").read_bytes()
        self.assertNotIn(b"\r\n", raw)
        raw.decode("ascii")
        text = raw.decode("ascii")
        self.assertNotIn("<img", text)
        self.assertNotIn("```math", text)
        self.assertNotIn("\\operatorname", text)
        equation_blocks = re.findall(r"(?m)^\$\$\n[^\n]+\n\$\$$", text)
        self.assertEqual(len(equation_blocks), 7)
        self.assertEqual(text.splitlines().count("$$"), 14)
        self.assertFalse((ROOT / "assets" / "equations").exists())

    def test_release_contains_only_dp_method_code(self):
        self.assertTrue((ROOT / "losses" / "dp_fy.py").is_file())
        source = (ROOT / "losses" / "dp_fy.py").read_text(encoding="utf-8")
        self.assertIn("def log_partition_dp", source)
        self.assertIn("def dp_fy_loss", source)
        self.assertIn("def exact_inclusion_marginals", source)

    def test_no_identity_or_cluster_artifacts(self):
        forbidden = (
            "/" + "gpfs/",
            "xj" + "tlu",
        )
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".zip"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, text, msg="{} contains {}".format(path, token))
        self.assertEqual(list(ROOT.rglob("*.out")), [])
        self.assertEqual(list(ROOT.rglob("*.err")), [])
        self.assertEqual(list(ROOT.rglob("*.pt")), [])


if __name__ == "__main__":
    unittest.main()
