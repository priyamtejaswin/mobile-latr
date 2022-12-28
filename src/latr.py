import pytorch_lightning as pl
from typing import Any
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from timm.models import vision_transformer
import torch
from typing import Dict, Optional, Union


class LaTrModel(pl.LightningModule):
    """
    Layout Aware Transformer --  <https://arxiv.org/pdf/2112.12494.pdf>
    """
    def __init__(self, config: dict) -> None:
        """
        Config:
            embed_size = 768
            vocab_size = 30522
        """
        super().__init__()

        # BERT embeds for all texts
        bert_config = BertConfig()
        self.bert_embeds = BertEmbeddings(bert_config)

        # Vision Transformer
        self.vit_model = getattr(
            vision_transformer,
            "vit_base_patch32_384"
        )(pretrained=True)

        # Transformer Encoder Decoder
        self.edt_model = torch.nn.Transformer(
            d_model=config["embed_size"],  # bert-base
            batch_first=True
        )

        # Vocab projection
        self.proj_vocab = torch.nn.Linear(
            in_features=config["embed_size"],
            out_features=config["vocab_size"] 
        )

        self.vocab_size = config["vocab_size"]
        self.id_sos = config["id_sos"]  # 101 for SOS, and 102 for EOS

    def forward(self, batch: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract question sentence embeddings.
        Extract image patch embeddings.
        Extract answer sentence embeddings.
        Pass to Transformer Encoder Decoder.
        """
        que_embeds = self.bert_embeds(batch["question_ids"])
        img_embeds = self.vit_model.forward_features(batch["image_tensors"])
        ans_embeds = self.bert_embeds(batch["answer_ids"])

        # TODO: confirm if the tgt_mask needs to be moved to a differnet device!
        tgt_mask = self.edt_model.generate_square_subsequent_mask(batch["answer_ids"].size(1))
        
        # TODO: add modality-type embedding to question and image
        # before concatenating embeddings!
        # <https://arxiv.org/pdf/2102.03334.pdf> -- Section 3.1
        z_input = torch.concat([que_embeds, img_embeds], dim=1)

        z_mask = torch.concat(
            [batch["question_atm"].bool(), 
             torch.zeros(img_embeds.shape[0], img_embeds.shape[1], dtype=torch.bool).to(self.device)],
            dim=1
        )

        z_output = self.edt_model(
            src=z_input,
            tgt=ans_embeds,
            tgt_mask=tgt_mask,  # The "square mask", so the model can't see the future.
            src_key_padding_mask=z_mask,  # To ignore padded tokens.
            tgt_key_padding_mask=batch["answer_atm"]  # To ignore padded tokens.
        )

        logits = self.proj_vocab(z_output)

        return logits

    def training_step(self, batch, batch_idx) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        output = self(batch)
        loss = torch.nn.functional.cross_entropy(
            input=output.permute(0, 2, 1), 
            target=batch["answer_ids"], 
            ignore_index=0
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        bs, tlen = batch["answer_ids"].size()
        target = batch["answer_ids"]
        output = torch.zeros([bs, tlen, self.vocab_size])
        generation = torch.zeros([bs, tlen], dtype=torch.int)

        # Focus on everything.
        # Ignore the future tokens.
        # Also, push to 
        batch["answer_atm"] = torch.zeros_like(batch["answer_atm"]).bool().to(self.device)

        for i in range(tlen):
            if i == 0:
                output[:, 0, self.id_sos] = 1.0
                generation[:, 0] = self.id_sos
            else:
                batch["answer_ids"] = generation.to(self.device)
                step = self(batch)
                output[:, i] = step[:, i]
                generation[:, i] = torch.argmax(output[:, i], dim=-1)

        loss = torch.nn.functional.cross_entropy(
            input=output.permute(0, 2, 1),
            target=target,
            ignore_index=0
        )
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=0.0001,
            weight_decay=0.01
        )