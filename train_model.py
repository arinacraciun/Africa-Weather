import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.data_processing.pipeline import run_preprocessing_pipeline
from src.training.datamodule import AfriCastDataModule
from src.models.lightning_gnn import WeatherLightningModule

def main():
    # 1. Prepare Data
    zarr_path, graph_path = run_preprocessing_pipeline('2023', '01')
    
    # 2. Initialize DataModule
    dm = AfriCastDataModule(zarr_path, graph_path, batch_size=1)
    dm.setup()
    
    # 3. Initialize Model
    # If you have 5 variables and 6 pressure levels:
    num_features = 5 * 6 

    model = WeatherLightningModule(
        in_channels=num_features, 
        hidden_channels=64, 
        out_channels=num_features
    )
    
    # 4. Trainer Configuration
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min')
    

    # 5. Early stopping prevents wasting time on flat loss
    early_stop = EarlyStopping(monitor="train_loss", patience=3, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=15, # Lowered for laptop sanity
        accelerator="cpu",
        precision="bf16-mixed", # Fixes the precision warning
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=1
    )
    
    # 6. Start Training
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()