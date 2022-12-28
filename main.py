from src.data_textvqa import TextVqaDataModule
from src.latr import LaTrModel
import pytorch_lightning as pl


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

    dm = TextVqaDataModule("./data", batch_size=32, workers=4)
    model = LaTrModel(config=CONFIG)

    trainer = pl.Trainer(
        max_epochs=5,
        logger=logger,
        accelerator="cpu"
    )

    trainer.fit(model, dm)