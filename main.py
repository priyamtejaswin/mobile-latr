from src.data_textvqa import TextVqaDataModule
from src.latr import LaTrModel
import pytorch_lightning as pl
import torch


CONFIG = {
    "embed_size": 768,
    "vocab_size": 30522,
    "id_sos": 101,
    "id_eos": 102
}


if __name__ == "__main__":
    logger = pl.loggers.TensorBoardLogger(
        CONFIG.get("log_dir", "logs"),
        name=f'simple',
    )

    dm = TextVqaDataModule("./data", batch_size=8, workers=8)
    model = LaTrModel(config=CONFIG)

    trainer = pl.Trainer(
        max_epochs=5,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10
    )

    trainer.fit(model, dm)