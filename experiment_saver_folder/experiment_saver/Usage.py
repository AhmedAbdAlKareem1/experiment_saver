from experiment_saver_folder.experiment_saver import ExperimentSaver, ExperimentConfig
#For More Info Check experiment_saver Repo
#https://github.com/AhmedAbdAlKareem1/experiment_saver
#on the top of your code, 
#cfg = ExperimentConfig(run_dir="runs/exp2")
#saver = ExperimentSaver(cfg, class_names=your class name/s)
    #Binary Saver
# 1) Create saver
saver = ExperimentSaver(
    config=ExperimentConfig(
        run_dir="runs/cat_dog_vgg16_exp001",
        patience=5,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    class_names=["Cat", "Dog"],
)

# 2) Train with callbacks
cfg = ExperimentConfig(run_dir="runs/exp2")
saver = ExperimentSaver(cfg, class_names=your class name/s)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=saver.callbacks(),
    verbose=1,
)

# 3) Save everything after training
saved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={
        "optimizer": "Adam",
        "lr": 1e-3,
        "dataset_path": r"path_to_dataset",
        "backbone": "VGG16",
        "image_size": [224, 224],
    },
)

print("Saved files:", saved_paths)



    #Mulity Class Saver
# 1) Create saver
saver = ExperimentSaver(
    config=ExperimentConfig(
        run_dir="runs/ham10000_exp001",
        patience=7,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        roc_average="macro",     # macro/micro/weighted will be computed when possible
        roc_multi_class="ovr",   # "ovr" recommended for multiclass ROC
    ),
    class_names=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
)

# 2) Train with callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=saver.callbacks(),
    verbose=1,
)

# 3) Save everything after training
saved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={
        "optimizer": "Adam",
        "lr": 1e-4,
        "dataset": "HAM10000",
        "image_size": [224, 224],
        "num_classes": 7,
    },
)

print("Saved files:", saved_paths)

