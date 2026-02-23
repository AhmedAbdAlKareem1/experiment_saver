🧪 experiment_saverA lightweight, safe utility for TensorFlow / Keras that automatically captures training artifacts and evaluation outputs. Stop rewriting boilerplate code and start comparing experiments with ease.✨ Highlights🚀 One-line Callbacks: Instant setup for CSVLogger, ModelCheckpoint, and EarlyStopping.💾 Comprehensive Saving: Automatically stores best + final models, history.json, and metrics.csv.📊 Auto-Evaluation: Computes ROC / AUC curves immediately after training.🛡️ Smart Handling: Safely manages common output shapes (Sigmoid, Binary Softmax, Multiclass).📝 Metadata Logging: JSON-safe config saving that handles non-serializable values gracefully.📂 Project StructureAll artifacts are organized within your specified run_dir/:CategoryFileDescriptionLogsmetrics.csvEpoch-by-epoch logs (loss, accuracy, AUC, etc.)history.jsonFull training history in JSON formatModelsbest_model.kerasThe "best" checkpoint based on your monitor metricfinal_model.kerasThe model state at the final epochEvaluationroc_fpr.npy / tpr.npyRaw ROC curve data for plottingroc_auc.jsonSummary of AUC scoresMetadataconfig.jsonExperiment hyperparameters and metadatamanifest.jsonA master index of all generated files⚙️ InstallationInstall from SourceBashgit clone https://github.com/AhmedAbdAlKareem1/experiment_saver.git
cd experiment_saver/experiment_saver_folder
pip install .
🚀 Quick Start1. Basic SetupPythonfrom experiment_saver_folder.experiment_saver import ExperimentSaver, ExperimentConfig

cfg = ExperimentConfig(
    run_dir="runs/cat_dog_vgg16_exp001",
    patience=5,
    monitor="val_loss",
    save_best_only=True
)

saver = ExperimentSaver(cfg, class_names=["Cat", "Dog"])
2. TrainingPass the generated callbacks into your model.fit() call to ensure metrics are logged correctly.Pythonhistory = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=saver.callbacks() # Essential for metrics.csv
)
3. Post-Training ExportPythonsaved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={
        "optimizer": "Adam",
        "lr": 1e-3,
        "backbone": "VGG16"
    }
)
🔧 Configuration DetailsThe ExperimentConfig class allows for fine-grained control:run_dir: The destination folder for all artifacts.monitor: Metric to track (e.g., "val_auc").roc_average: Strategy for multiclass ("macro", "micro", or "weighted").positive_class_index: For binary softmax models (defaults to 1).💡 Troubleshooting[!TIP]Missing metrics.csv? > This usually happens if you forgot to pass callbacks=saver.callbacks() into model.fit(), or if the training crashed before the first epoch completed.
