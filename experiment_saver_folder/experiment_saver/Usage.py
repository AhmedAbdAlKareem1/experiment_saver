# 1) Create saver
from experiment_saver_folder.experiment_saver import ExperimentSaver, ExperimentConfig

saver = ExperimentSaver(
    #run_dir is the place you would save the model
    config=ExperimentConfig(run_dir="runs/cat_dog_vgg16_exp001", patience=5),
    class_names=["Cat", "Dog"]
)

# 2) Train with callbacks
history = model.fit(
    your_train_set,
    validation_data=your_val_set,
    epochs=5,
)

# 3) Save everything after training
saved_paths = saver.save_after_fit(
    model=your_model,
    history=history,
    val_ds=your_val_set,
    extra_config={
        "optimizer": "you_optimizer",
        "lr": 1e-3,#learning rate
        "dataset_path": r"path_to_dataset"
    }
)

print("Saved files:", saved_paths)