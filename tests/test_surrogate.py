"""Unit tests for neural surrogate model and active learning candidate screening."""
import unittest
from forge.core.models import DuctSpec, PhysicsConditions
from forge.surrogate.model import DuctFlowSurrogate
from forge.surrogate.active_learning import screen_candidates


class SurrogateTests(unittest.TestCase):
    def setUp(self):
        self.model = DuctFlowSurrogate.create_pretrained()
        self.spec = DuctSpec(
            inlet_width=120, inlet_height=120, outlet_width=140, outlet_height=45,
            length=100, wall=2.0, guide_vanes=1
        )
        self.conds = PhysicsConditions(volumetric_flow_cfm=150.0)

    def test_surrogate_prediction(self):
        pred = self.model.predict_spec(self.spec, self.conds)
        self.assertIn("predicted_delta_p_pa", pred)
        self.assertIn("predicted_uniformity_index", pred)
        self.assertGreater(pred["predicted_delta_p_pa"], 0.0)
        self.assertTrue(0.0 <= pred["predicted_uniformity_index"] <= 1.0)

    def test_active_learning_screening(self):
        candidates = [
            self.spec,
            DuctSpec(inlet_width=100, inlet_height=80, outlet_width=120, outlet_height=50, length=120, wall=2.0),
        ]
        screened = screen_candidates(candidates, self.conds, self.model, target_max_dp_pa=100.0)
        self.assertGreaterEqual(len(screened), 1)
        for cand, pred_dp in screened:
            self.assertLessEqual(pred_dp, 100.0)


if __name__ == "__main__":
    unittest.main()

