import unittest
import numpy as np
import sys
import os

# Add root to path to import world.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import world

class TestWorld(unittest.TestCase):
    def setUp(self):
        self.field = world.SparseField(latent_dim=4)

    def test_encode_generic(self):
        """Test generic vector encoding."""
        data = [1.0, 2.0, 3.0, 4.0]
        obs_list = world.encode(data, modality="generic", t=0.5)
        self.assertEqual(len(obs_list), 1)
        self.assertTrue(np.allclose(obs_list[0].value, data))
        self.assertTrue(np.allclose(obs_list[0].pos, [0.5, 0.5, 0.5, 0.5]))

    def test_encode_image(self):
        """Test 2D image encoding."""
        # 2x2 image
        data = np.array([
            [[1], [2]],
            [[3], [4]]
        ])
        obs_list = world.encode(data, modality="image", t=0.0)
        self.assertEqual(len(obs_list), 4) # 2x2 = 4 pixels
        # Check normalization
        self.assertEqual(obs_list[0].pos[0], 0.0) # x=0/2
        self.assertEqual(obs_list[-1].pos[1], 0.5) # y=1/2

    def test_observe_and_query(self):
        """Test observing a point and querying it back."""
        obs = world.Observation(pos=[0.1, 0.1, 0.1, 0.0], value=[1, 0, 0, 0])
        self.field.observe([obs])
        
        # Verify stored
        self.assertEqual(len(self.field.positions), 1)
        
        # Query exact location
        val, cert = self.field.query([0.1, 0.1, 0.1, 0.0])
        self.assertTrue(np.allclose(val, [1, 0, 0, 0]))
        self.assertGreater(cert, 0.0)
        
        # Query far away (should be 0)
        val_far, cert_far = self.field.query([0.9, 0.9, 0.9, 0.0])
        self.assertLess(cert_far, cert)

    def test_multiple_observations_blending(self):
        """Test that nearby observations blend."""
        obs1 = world.Observation(pos=[0.5, 0.5, 0.5, 0.0], value=[1, 0, 0, 0])
        obs2 = world.Observation(pos=[0.51, 0.5, 0.5, 0.0], value=[0, 1, 0, 0])
        
        # Should blend due to proximity
        self.field.observe([obs1, obs2])
        
        # Query midpoint
        val, cert = self.field.query([0.505, 0.5, 0.5, 0.0])
        # Should be mix of both
        self.assertGreater(val[0], 0.0)
        self.assertGreater(val[1], 0.0)

    def test_compress(self):
        """Test compression of redundant points."""
        # Add two identical points
        obs1 = world.Observation(pos=[0.5, 0.5, 0.5, 0.0], value=[1, 1, 1, 1])
        obs2 = world.Observation(pos=[0.501, 0.5, 0.5, 0.0], value=[1, 1, 1, 1])
        
        # Observe both at once so they are added separately (tree updates after loop)
        self.field.observe([obs1, obs2])
        
        initial_count = len(self.field.positions)
        self.assertEqual(initial_count, 2, "Should have 2 points before compression")
        
        self.field.compress(small_radius=0.05)
        final_count = len(self.field.positions)
        
        self.assertLess(final_count, initial_count)

    def test_causal_learning(self):
        """Test learning of temporal correlations."""
        # Event A at t=0
        obs1 = world.Observation([0.5, 0.5, 0.5, 0.0], [1, 0, 0, 0])
        # Event B at t=0.1
        obs2 = world.Observation([0.5, 0.5, 0.5, 0.1], [1, 0, 0, 0])
        
        self.field.observe([obs1, obs2])
        self.field.learn_causal_edges(dt=0.2)
        
        # Should find correlation
        self.assertTrue(len(self.field.edges) > 0)

    def test_respond_function(self):
        """Test the high-level respond function."""
        obs = [world.Observation([0.5,0.5,0.5,0.0], [1,-1,0,0])]
        val, cert = world.respond(self.field, obs)
        
        # Value aligned output should be non-negative
        self.assertTrue(np.all(val >= 0))

if __name__ == '__main__':
    unittest.main()
