# VisionSpeak

VisionSpeak converts visible lip movements and hand gestures into text.
Lip reading recognises eight isolated words: **hello, hi, yes, no, good, bad, thanks, bye**.

**Workflow:** prepare environments -> record words -> train models and save checkpoints -> evaluate -> run the combined webcam application.

Run terminal commands from the **project root**. Examples below use Windows PowerShell.

## 1. Prepare the Environments

Use Python 3.11. Keep lip-reading/MediaPipe and YOLO in separate environments because they use different OpenCV packages.

~~~powershell
py -3.11 -m venv .venv
$lipPython = "./.venv/Scripts/python.exe"
& $lipPython -m pip install -r requirements.txt
& $lipPython -m ipykernel install --user --name visionspeak-lip --display-name "VisionSpeak Lip"

py -3.11 -m venv .venv-yolo
$yoloPython = "./.venv-yolo/Scripts/python.exe"
& $yoloPython -m pip install -r requirements-yolo.txt
& $yoloPython -m ipykernel install --user --name visionspeak-yolo --display-name "VisionSpeak YOLO"
~~~

Open notebooks in VS Code or Jupyter. Select **VisionSpeak Lip** for recording, lip training, evaluation, and the combined notebook; select **VisionSpeak YOLO** for the YOLO notebook. An existing compatible environment such as NUS may be used instead: set `$lipPython` to its Python executable and choose its notebook kernel. On Linux/macOS, use `bin/python` instead of `Scripts/python.exe`.

## 2. Record Custom Words

Open [data/record_custom_word_dataset.ipynb](data/record_custom_word_dataset.ipynb).

1. Run setup, then set a unique anonymous `SPEAKER_ID` for each person.
2. Set `NUM_ROUNDS` (default 20 per word) and `SAVE_RAW_VIDEO = True` for MP4 evaluation. Obtain consent for saved face video.
3. Run the helper and collection cells. During review, **S** saves, **R** retries, **P** replays, and **Q** pauses. Rerun collection to resume.
4. Run the audit cell to check the saved recordings.

Recordings are saved under `data/custom_words/`: mouth crops in `clips/`, videos in `raw/`, and labels/paths in `manifest.csv`. No audio is saved. Keep the face-landmarker asset at `processing/face_landmarker.task`.

[data/prepare_custom_word_dataset.py](data/prepare_custom_word_dataset.py) loads and splits the recordings automatically when training starts; there is no separate preprocessing command.

## 3. Train Individual Lip Models and Save Checkpoints

Choose a recorded speaker to exclude from training. Each command below trains **one** model; run only the ones you need.

~~~powershell
$testSpeaker = "speaker_test_samples"
$epochs = 100
$runDir = "checkpoints/custom_run"

# Word classifier: 3D CNN + BiGRU
& $lipPython -m models.LBY.train_custom_words --architecture bigru --speaker-holdout $testSpeaker --all-training-data --epochs $epochs --lr 1e-3 --batch-size 32 --checkpoint "$runDir/word_bigru_ep${epochs}.pt"

# Word classifier: 3D CNN + Transformer
& $lipPython -m models.LBY.train_custom_words --architecture transformer --pooling attention --speaker-holdout $testSpeaker --all-training-data --epochs $epochs --lr 1e-3 --batch-size 32 --checkpoint "$runDir/word_transformer_attention_ep${epochs}.pt"

# Character-sequence model: 3D CNN + BiGRU + CTC
& $lipPython -m models.valente.train_custom_words --architecture bigru --speaker-holdout $testSpeaker --all-training-data --epochs $epochs --lr 1e-4 --batch-size 16 --checkpoint "$runDir/ctc_bigru_ep${epochs}.pt"

# Character-sequence model: 3D CNN + Transformer + CTC
& $lipPython -m models.valente.train_custom_words --architecture transformer --speaker-holdout $testSpeaker --all-training-data --epochs $epochs --lr 1e-4 --batch-size 16 --checkpoint "$runDir/ctc_transformer_ep${epochs}.pt"
~~~

Training automatically creates the checkpoint directory and writes the `.pt` file; no additional export step is needed. These commands use seed 42 and cosine learning-rate decay. Reduce batch size if GPU memory is insufficient.

With `--all-training-data`, every non-test-speaker clip trains and the **final epoch** is saved. Omit that flag to reserve validation clips and save the best validation checkpoint instead. Always keep `--speaker-holdout` for speaker-independent testing, and use a different run directory to preserve earlier checkpoints.

For the full epoch sweep, repeat the four commands with `$epochs = 50`, `100`, and `200`. The sweep expects the exact `<architecture>_ep<epoch>.pt` filenames shown above; combined live inference instead identifies models from checkpoint metadata.

## 4. Evaluate Lip Models

**One checkpoint on recorded MP4s:**

~~~powershell
& $lipPython evaluation/lip-reading/evaluate_custom_words.py --checkpoint "$runDir/ctc_bigru_ep100.pt" --source mp4 --speakers $testSpeaker --output-csv "reports/custom_run_predictions.csv"
~~~

The report directory is created automatically. Replace the checkpoint to evaluate another model. Use `--source npz` for saved mouth crops; this bypasses raw-video mouth extraction.

**One video, prediction only:**

~~~powershell
& $lipPython evaluation/lip-reading/evaluate_custom_words.py --checkpoint "$runDir/ctc_bigru_ep100.pt" --video "path/to/recording.mp4"
~~~

**Accuracy tables and confusion-matrix figures across models/epochs:** open [offline_epoch_sweep.ipynb](evaluation/lip-reading/offline_epoch_sweep.ipynb). Set all four `CHECKPOINT_DIRS` values to your run directory, select `CLIPS_DIR = 'data/custom_words/raw/<test-speaker>'`, and set `EPOCH_BUDGETS = (100,)` or `(50, 100, 200)` to match available checkpoints. Set `ALLOW_SPEAKER_MISMATCH = False` for held-out evaluation, then run the cells. Reports and plots are saved under `reports/retrain/`.

The terminal equivalent selects recordings from the manifest:

~~~powershell
& $lipPython evaluation/lip-reading/evaluate_epoch_sweep.py --checkpoint-dir $runDir --speakers $testSpeaker --epochs 100
~~~

For a comparison of two selected checkpoints with additional statistics and plots, use [compare_lip_models.py](evaluation/lip-reading/compare_lip_models.py) with `--ctc-checkpoint` and `--transformer-checkpoint`.

Accuracy excludes clips that fail preprocessing or inference, so review rejection counts. CTC evaluation ranks the eight allowed words; greedy character-decoding accuracy reported during training is a different measure.

## 5. Train and Evaluate Gesture Models

Gesture training uses labelled **images**, separate from the custom-word videos.

| Model | Training and evaluation workflow | Checkpoints |
| --- | --- | --- |
| YOLOv8n | [main_yolo8n.ipynb](models/Lawrence/main_yolo8n.ipynb), using the YOLO kernel | `checkpoints/retrain/yolov8n50.pt`, `yolov8n100.pt`, `yolov8n150.pt` |
| MediaPipe + LSTM | [sign_language_workflow.ipynb](models/SX/sign_language_workflow.ipynb) | `checkpoints/retrain/Sign_Language_MediaPipe_LSTM_<epochs>.pt` |
| MediaPipe + Transformer | [sign_language_Tranformer_workflow.ipynb](models/SX/sign_language_Tranformer_workflow.ipynb) | `checkpoints/retrain/Sign_Language_MediaPipe_Transformer_<epochs>.pt` |

For YOLO, first update `models/Lawrence/data.yaml`: its dataset paths currently refer to another computer. Run the notebook from `models/Lawrence`, train separate 50/100/150-epoch runs, and copy each run's printed `weights/best.pt` to the corresponding checkpoint filename above. The final evaluation/plotting cells compare them using `test_selfie`.

For Mediapipe+LSTM & Transformer, provide per-class image folders under `SuXin_TEMP/data/{train,valid,test}/<class>/`. Set `EPOCHS` to 50, 100, or 150 in the relevant training cell; each run saves its best-validation checkpoint automatically. The **final offline evaluation cell in the Transformer notebook evaluates both LSTM and Transformer checkpoints** on the same webcam images as YOLO. Confusion matrices are saved under `models/SX/reports/`; YOLO's figure is under `models/Lawrence/reports/`.

**Compatibility:** Mediapipe+LSTM & Transformer training/cache extraction and notebook webcam cells still use legacy `mp.solutions`, which is unavailable in the default MediaPipe 0.10.35 environment. Those cells require a compatible separate training environment or Tasks API migration. The combined application's Mediapipe+LSTM & Transformer worker already uses the Tasks API. Static-image gesture experiments do not establish recognition of real signing motion.

## 6. Run the Combined Application

Open [combinedVisionSpeak.ipynb](combinedVisionSpeak.ipynb) with the lip-reading kernel.

1. Run the launcher and environment-setup cells. Restart the kernel if dependencies were installed.
2. Replace the configuration cell's checkpoint paths with real files. For the training example above:

~~~python
CONFIG = CombinedConfig(
    project_root=PROJECT_ROOT,
    lip_python=LIP_PYTHON,
    gesture_python=GESTURE_PYTHON,
    lip_checkpoint="checkpoints/retrain/word_transformer_attention_ep200.pt",
    gesture_checkpoint="checkpoints/retrain/yolov8n50.pt",
)
~~~

3. Focus the OpenCV window: **Space** records one lip word, **M** switches modes, and **Q** quits. The summary cell shows how the session ended.

To use supplied lip weights without retraining, select `checkpoints/retrain/word_transformer_attention_ep200.pt` or `checkpoints/retrain/ctc_transformer_ep200.pt`. YOLO gesture checkpoints run in `.venv-yolo`; Mediapipe+LSTM & Transformer checkpoints run in the lip environment. Checkpoint metadata selects CTC versus word classification and YOLO versus Mediapipe+LSTM & Transformer automatically. Set `LIP_PYTHON` / `GESTURE_PYTHON` to interpreter paths if using custom environment locations.

Earlier notebook experiments remain under [archive/kaggle_lip_reading](archive/kaggle_lip_reading/README.md).

## 7. Credits / Project Contribution
| **Official Full Name** | **Student ID (MTech Applicable)** | **Work Items (Who Did What)** | **Email (Optional)** |
| ---------------------- | --------------------------------- | ----------------------------- | -------------------- |
| Yew Hwee Peng | A0026237L | [Train, validate, test and evaluate the YOLOv8n models](./models/Lawrence/main_yolo8n.ipynb), [Detect hand-gestures on real-time Webcam](./models/Lawrence/gesture_detector.py), [Selfie of hand-gesture for the 8-words vocabulary using webcam for neutral evaluation](./models/Lawrence/selfie_hand_gesture.ipynb), [Generated YOLOv8n checkpoint for hand-gesture pipeline](./checkpoints/yolov8n_custom.pt), Analysis of YOLOv8n confusion matrix results for 50, 100 and 150 epoch | [yew_hwee_peng@u.nus.edu](mailto:yew_hwee_peng@u.nus.edu) |
| Su Xin | A0339793J | Built, trained, validated, and evaluated the [MediaPipe + LSTM](./models/SX/sign_language_workflow.ipynb) and [MediaPipe + Transformer](./models/SX/sign_language_Tranformer_workflow.ipynb) models. Live evaluation of detecting hand-gestures with Webcam for [MediaPipe + LSTM](./models/SX/sign_language_workflow.ipynb) and [MediaPipe + Transformer](./models/SX/sign_language_Tranformer_workflow.ipynb) models. [Generated confusion matrices](./models/SX/sign_language_Tranformer_workflow.ipynb) and hand-gesture pipeline checkpoints for both models. Conducted evaluations and class-level error analysis on the MediaPipe + LSTM and MediaPipe + Transformer matrix results for 50, 100 and 150 epochs. | [su.xin@u.nus.edu](mailto:su.xin@u.nus.edu) |
| Valente Lam | A0339779B | **Model development:** Implemented the [LipNet-style spatiotemporal CNN + BiGRU + CTC model](./models/valente/model.py) and contributed its training, prediction and checkpoint workflows.<br>**Data preparation:** Sourced Roboflow and Kaggle datasets; developed [landmark-based mouth cropping](./processing/video_loader.py), [grayscale preprocessing and frame padding](./processing/preprocessor.py).<br>**Evaluation:** Added CTC confusion-matrix and [webcam model-comparison functionality](./evaluation/lip-reading/webcam_lip_model_comparison.ipynb); extracted and reorganised the [offline evaluation workflows](./evaluation/lip-reading/).<br>**System integration:** Extended the [combined application](./combinedVisionSpeak.ipynb) with checkpoint-based CTC/word-classification switching, automatic YOLO/Media Pipe LSTM/Transformer backend selection, environment routing and checkpoint validation, integrating the models into one webcam system.<br>**Compatibility and maintenance:** Migrated [Mediapipe+LSTM & Transformer live inference](./inference/sx_worker.py) and [shared feature extraction](./models/SX/features.py) to MediaPipe's Tasks API; added regression tests; reorganised datasets, evaluation scripts and checkpoints; simplified setup and usage documentation.<br>**Report:** Contributed to the project aim, scope and objectives, evaluation methodology, and challenges and mitigation sections. | [valente.lam@u.nus.edu](mailto:valente.lam@u.nus.edu) |
| Lee Boon Yao | A0339757J | Alignment algorithm of [Kaggle Lip Reading Dataset](./archive/kaggle_lip_reading/align_videos.py), [Lip reading script data collection script](./data/record_custom_word_dataset.ipynb), [Word Level Detection for Lip Reading](./models/LBY/), [Initial Transformer Implementation of Lip and Hand](./models/transformer/), Environment requirement seperation for [Lip](./requirements.txt) and [Hand](./requirements-yolo.txt), [Batch training for lip reading models](./scripts/retrain_lip_models.sh), [Initial combination of lip and hand gesture architecture in jupyter notebook](./combinedVisionSpeak.ipynb), [Validation and Verification of Lip reading](./evaluation/lip-reading/), Analysis of Lip Reading result in report, System Architecture implementation and Keyboard command modality routing | [lee.boonyao@u.nus.edu](mailto:lee.boonyao@u.nus.edu) |
