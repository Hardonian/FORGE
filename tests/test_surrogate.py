"""Unit tests for Physics-Informed Neural Surrogate model, training, and active learning."""
import unittest
from forge.core.models import DuctSpec, PhysicsConditions
from forge.surrogate.model import DuctFlowSurrogate
from forge.surrogate.active_learning import screen_candidates, rank_candidates_by_acquisition


class SurrogateTests(unittest.TestCase):
    def setUp(self):
        self.model = DuctFlowSurrogate.create_pretrained()
        self.spec = DuctSpec(
            inlet_width=120, inlet_height=120, outlet_width=140, outlet_height=45,
            length=100, wall=2.0, guide_vanes=1
        )
        self.conds = PhysicsConditions(volumetric_flow_cfm=150.0)

    def test_surrogate_prediction_and_mc_dropout_uncertainty(self):
        pred = self.model.predict_spec(self.spec, self.conds)
        self.assertIn("predicted_delta_p_pa", pred)
        self.assertIn("predicted_uniformity_index", pred)
        self.assertIn("predicted_outlet_velocity_m_s", pred)
        self.assertGreater(pred["predicted_delta_p_pa"], 0.0)
        self.assertTrue(0.0 <= pred["predicted_uniformity_index"] <= 1.0)

        unc = self.model.predict_with_uncertainty(self.spec, self.conds, num_samples=12)
        self.assertIn("std_delta_p_pa", unc)
        self.assertGreaterEqual(unc["std_delta_p_pa"], 0.0)

    def test_surrogate_training_reduces_loss(self):
        specs = [
            self.spec,
            DuctSpec(inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50, length=120, wall=2.0),
            DuctSpec(inlet_width=120, inlet_height=120, outlet_width=160, outlet_height=60, length=80, wall=2.0, guide_vanes=2),
        ]
        metrics = self.model.train_on_specs(specs, self.conds, epochs=30, lr=1e-2)
        self.assertLess(metrics["final_loss"], metrics["initial_loss"])

    def test_active_learning_screening_and_lcb_acquisition(self):
        candidates = [
            self.spec,
            DuctSpec(inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50, length=120, wall=2.0),
        ]
        screened = screen_candidates(candidates, self.conds, self.model, target_max_dp_pa=100.0)
        self.assertGreaterEqual(len(screened), 1)
        for _, pred_dp in screened:
            self.assertLessEqual(pred_dp, 100.0)

        ranked = rank_candidates_by_acquisition(candidates, self.conds, self.model)
        self.assertEqual(len(ranked), 2)
        self.assertLessEqual(ranked[0]["lcb_acquisition_score"], ranked[1]["lcb_acquisition_score"])


if __name__ == "__main__":
    unittest.main()
