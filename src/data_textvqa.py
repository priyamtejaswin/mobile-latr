from torch.utils.data import Dataset, DataLoader
import pyarrow as pa
import io
from PIL import Image
from collections import Counter
from torchvision.transforms import PILToTensor, Resize, Normalize
from torchvision.transforms import InterpolationMode
import torch
from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq
from transformers import BertTokenizer
from typing import Dict, Optional


class TextVqaDataset(Dataset):
    def __init__(self, path: str, split: str, tokenizer: PreTrainedTokenizer, size: int=384, rescale: bool=True):
        self.rescale = rescale
        self.split = split
        assert self.split in ("train", "val", "test"), f"Invalid split arg {self.split} !"

        self.table = pa.ipc.RecordBatchFileReader(
            pa.memory_map(path, "r")
        ).read_all()

        self.pil_to_tensor = PILToTensor()
        self.transforms = torch.nn.Sequential(
            Resize([size, size], InterpolationMode.BICUBIC),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.table)

    def bytes2image(self, b: bytes):
        raw = io.BytesIO(b)
        raw.seek(0)
        return Image.open(raw).convert("RGB")

    def pil2tensor(self, img: Image):
        tensor = self.pil_to_tensor(img)
        if self.rescale:
            tensor = tensor / 255.0
        return tensor.unsqueeze(0)

    def __getitem__(self, index):
        img_bytes = self.table["image"][index].as_py()
        image = self.bytes2image(img_bytes)
        img_tensor = self.pil2tensor(image)
        img_ready = self.transforms(img_tensor)

        question = self.table["question"][index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index].as_py()
            topans = [Counter(answers).most_common(1)[0][0]]
        else:
            topans = list()

        return {
            "img_tensor": img_ready,
            "q_text": [question],
            "ans_text": topans
        }

    def collate(self, batch) -> Dict[str, Optional[torch.Tensor]]:
        q = []
        ans = []
        img = []

        for row in batch:
            q.append(row["q_text"][0])
            img.append(row["img_tensor"])

            if self.split != "test":
                ans.append(row["ans_text"][0])

        image_tensors = torch.concatenate(img, dim=0)
        question_tensors = self.tokenizer(q, return_tensors="pt", padding="longest")

        if self.split != "test":
            answer_tensors = self.tokenizer(ans, return_tensors="pt", padding="longest")
        else:
            answer_tensors = None

        return {
            "image_tensors": image_tensors,
            "question_ids": question_tensors["input_ids"],
            "question_atm": question_tensors["attention_mask"],
            "answer_ids": None if answer_tensors is None else answer_tensors["input_ids"],
            "answer_atm": None if answer_tensors is None else answer_tensors["attention_mask"]
        }


if __name__ == "__main__":
    path = "data/textvqa_mini_train.arrow"
    split = "train"

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = TextVqaDataset(path, split, bert_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=train_dataset.collate)

    for batch in train_loader:
        print(batch)