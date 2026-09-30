import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

from models.SX.features import feature_vector, extract_rgb
from models.SX.inference import load_sx_checkpoint
from inference.sx_worker import SXDetector


def landmarks(count, value):
    return NS(landmark=[NS(x=value, y=value+1, z=value+2) for _ in range(count)])


class SXTests(unittest.TestCase):
    def test_order_and_zero_filling(self):
        first, second = landmarks(21,1), landmarks(21,4)
        vector = feature_vector([first,second],landmarks(33,7),use_body_pose=True)
        self.assertEqual(vector.shape,(225,))
        self.assertEqual(vector.dtype,np.float32)
        np.testing.assert_equal(vector[:3],[1,2,3])
        np.testing.assert_equal(vector[63:66],[4,5,6])
        np.testing.assert_equal(vector[126:129],[7,8,9])
        missing = feature_vector([first],None,use_body_pose=True)
        self.assertTrue(np.all(missing[63:]==0))
        self.assertFalse(feature_vector(None,None).any())

    def test_existing_checkpoints_forward_and_no_hand_gate(self):
        root = Path(__file__).resolve().parents[1]
        for suffix,size in [('',126),('_body',225)]:
            path=root/'checkpoints'/f'sx_hand_sign_cached_both_hands{suffix}.pt'
            model, labels, config = load_sx_checkpoint(path,torch.device('cpu'))
            self.assertEqual(config.input_size,size)
            with torch.no_grad():
                output=model(torch.zeros(1,1,size))
            self.assertEqual(output.shape,(1,18))
            detector=SXDetector(path,device='cpu')
            detector.hands,detector.pose=Mock(),Mock()
            detector.model=Mock()
            with patch('inference.sx_worker.extract_rgb',return_value=(np.zeros(size,dtype=np.float32),None)):
                label,confidence,render=detector.predict_frame(np.zeros((50,100,3),dtype=np.uint8))
            self.assertEqual(label,'No hand detected')
            detector.model.assert_not_called()

    def test_incompatible_metadata_and_sequences_rejected(self):
        for checkpoint, message in [({'input_size':225,'two_hands':True,'use_body_pose':False},'metadata'),
                                    ({'sequence_length':2},'sequence_length')]:
            with patch('models.SX.inference.torch.load',return_value=checkpoint):
                with self.assertRaisesRegex(ValueError,message):
                    load_sx_checkpoint('unused',torch.device('cpu'))

    def test_shared_image_extraction_matches_vector(self):
        hands=Mock(); hands.process.return_value=NS(multi_hand_landmarks=[landmarks(21,1)])
        pose=Mock(); pose.process.return_value=NS(pose_landmarks=landmarks(33,2))
        vector,_=extract_rgb(np.zeros((50,100,3),np.uint8),hands=hands,pose=pose,use_body_pose=True)
        np.testing.assert_equal(vector,feature_vector([landmarks(21,1)],landmarks(33,2),use_body_pose=True))

    def test_training_and_prediction_image_features_agree(self):
        from models.SX.train import extract_hand_landmarks_from_image
        from models.SX.predict import extract_landmarks_from_image
        vector=feature_vector([landmarks(21,1)],landmarks(33,2),use_body_pose=True)
        with patch('cv2.imread',return_value=np.zeros((20,20,3),np.uint8)), \
             patch('models.SX.features.extract_rgb',return_value=(vector,[landmarks(21,1)])):
            train=extract_hand_landmarks_from_image(Path('unused'),use_body_pose=True)
            predict=extract_landmarks_from_image('unused',use_body_pose=True)
        np.testing.assert_equal(train,predict)

    def test_low_confidence_and_body_missing(self):
        root=Path(__file__).resolve().parents[1]
        detector=SXDetector(root/'checkpoints/sx_hand_sign_cached_both_hands_body.pt',device='cpu')
        detector.hands,detector.pose=Mock(),Mock()
        detector.model=Mock(return_value=torch.zeros(1,18))
        vector=feature_vector([landmarks(21,1)],None,use_body_pose=True)
        with patch('inference.sx_worker.extract_rgb',return_value=(vector,[landmarks(21,1)])):
            label,confidence,render=detector.predict_frame(np.zeros((240,320,3),np.uint8))
        self.assertEqual(label,'Low confidence')
        self.assertAlmostEqual(confidence,1/18)
        self.assertEqual(detector.model.call_args.args[0].shape,(1,1,225))
        self.assertEqual(render().shape,(240,320,3))

    def test_camera_open_failure_closes_landmark_resources(self):
        root=Path(__file__).resolve().parents[1]
        detector=SXDetector(root/'checkpoints/sx_hand_sign_cached_both_hands_body.pt',device='cpu')
        hands,pose=Mock(),Mock()
        with patch('processing.mediapipe_compat.mp.solutions.hands.Hands') as hand_factory, \
             patch('processing.mediapipe_compat.mp.solutions.pose.Pose') as pose_factory, \
             patch('cv2.VideoCapture') as camera, patch('cv2.destroyAllWindows'):
            hand_factory.return_value.__enter__.return_value=hands
            pose_factory.return_value.__enter__.return_value=pose
            camera.return_value.isOpened.return_value=False
            from inference.gesture_live import run_gesture_live
            self.assertEqual(run_gesture_live(detector,title='SX'),'error')
            camera.return_value.release.assert_called_once()
            hand_factory.return_value.__exit__.assert_called_once()
            pose_factory.return_value.__exit__.assert_called_once()


if __name__=='__main__':
    unittest.main()
