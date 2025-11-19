import logging
import torch
from transformers.data.data_collator import *
from inference.ICL import TASK_PROMT, Constrained_PROMPT
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable, Tuple
from PIL import Image
import os

logger = logging.getLogger(__name__)

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor  
    pixel_values: torch.Tensor  
    a_input_ids: torch.Tensor  

@dataclass
class MultiModalDataCollator:
    processor: Any                         
    max_prompt_len: Optional[int] = None   
    max_ans_len: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    pad_to_multiple_of: Optional[int] = None
    truncation_side: str = "left"          
    include_eos_in_labels: bool = True     
    inference: bool = False          

    def __post_init__(self):
        if hasattr(self.processor, "tokenizer"):
            try:
                self.processor.tokenizer.truncation_side = self.truncation_side
            except Exception:
                pass

    def _flatten_instances(self, batch: List[Dict]) -> List[Tuple[str, str, str]]:
        flat: List[Tuple[str, str, str]] = []
        for instance in batch:
            p, a, u = instance["prompt"], instance["answer"], instance["pic"]
            if isinstance(p, (list, tuple)):
                assert isinstance(a, (list, tuple)) and isinstance(u, (list, tuple))
                assert len(p) == len(a) == len(u)
                flat.extend(zip(p, a, u))
            else:
                flat.append((p, a, u))
        return flat
    
    def build_qaimage(self, q_text, a_text, image_path):
        messages = [
                {"role": "user",
                "content": [
                    {"type": "text", "text": q_text},
                    {"type": "image"}
                ]},
        ]
        # prompt = self.processor.tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )  
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # stop = input(prompt)
        # stop = input(prompt_old)
        # stop = input(prompt_old==prompt)
        raw_image = Image.open(image_path) 
        inputs = self.processor(raw_image, prompt, return_tensors="pt")

        a_input_ids = self.processor.tokenizer(
            a_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        # stop = input(a_input_ids)
        res = QaImageOutput(
            q_input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            a_input_ids=a_input_ids,
        )
        return res

    def convert_one_piece(
            self,
            q_input_ids: torch.Tensor,
            a_input_ids: torch.Tensor,
        ):
        input_ids = torch.concat(
            [
                q_input_ids,  
                a_input_ids, 
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        ) 
        
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.label_pad_token_id),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        ) 

        return input_ids, labels
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        samples = self._flatten_instances(batch)
        texts_full: List[str] = []
        texts_prompt_only: List[str] = []
        images: List[Any] = []
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []
        
        for prompt, answer, img_path in samples:
            qaimage_output = self.build_qaimage(prompt, answer, img_path)
            
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )           
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)
        
        max_input_len = max(max_input_len_list)           
        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.label_pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        out = {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }
        # stop = input(out)
        return out
        
        # # stop = input(texts_full)
        # # stop = input(texts_prompt_only)
        # tok = self.processor.tokenizer
        # full_lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts_full]
        # prompt_only_lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts_prompt_only]
        # ans_lens = [max(fl - pl, 0) for fl, pl in zip(full_lens, prompt_only_lens)]
        # if self.max_ans_len is not None:
        #     ans_lens = [min(l, self.max_ans_len) for l in ans_lens]
        # # 对文本做一致的 batch tokenization（padding/truncation 保证多样本能转为张量）
        # max_total_len = None
        # if self.max_prompt_len is not None and self.max_ans_len is not None:
        #     max_total_len = self.max_prompt_len + self.max_ans_len + 336
        # elif self.max_ans_len is not None:
        #     max_total_len = self.max_ans_len + 336 # 也可以视情况改为 None

        # enc = self.processor(
        #     text=texts_full,
        #     images=images,
        #     return_tensors=self.return_tensors,
        #     padding=True,
        #     truncation=True,
        #     max_length=max_total_len,
        # )

        # if "input_ids" in enc and "attention_mask" in enc:
        #     input_ids = enc["input_ids"]  # 已经是 tensor (B, T)
        #     attention_mask = enc["attention_mask"]
        # else:
        #     text_enc = self.processor.tokenizer(
        #         texts_full,
        #         padding=True,
        #         truncation=True,
        #         max_length=max_total_len,
        #         return_tensors=None,  # 得到 lists 而非 tensors
        #         add_special_tokens=True,
        #     )
        #     input_ids_list = text_enc["input_ids"]
        #     attention_mask_list = text_enc["attention_mask"]
        #     input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        #     attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        # B, T = input_ids.shape

        # labels = input_ids.clone()
        # for i in range(B):
        #     seq_len = int(attention_mask[i].sum().item())
        #     keep = ans_lens[i]
        #     if self.max_ans_len is not None:
        #         keep = min(keep, self.max_ans_len)
        #     if not self.include_eos_in_labels and keep > 0:
        #         keep = max(keep - 1, 0)
        #     prompt_len_kept = max(seq_len - keep, 0)
        #     labels[i, :prompt_len_kept] = self.label_pad_token_id
        #     # labels[i, seq_len:] = self.label_pad_token_id

        # if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 1:
        #     multiple = self.pad_to_multiple_of
        #     pad_len = ((T + multiple - 1)//multiple)*multiple - T
        #     if pad_len > 0:
        #         pad_ids = input_ids.new_full((B, pad_len), tok.pad_token_id)
        #         pad_mask = attention_mask.new_zeros((B, pad_len))
        #         pad_lbls = labels.new_full((B, pad_len), self.label_pad_token_id)
        #         input_ids = torch.cat([input_ids, pad_ids], dim=1)
        #         attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
        #         labels = torch.cat([labels, pad_lbls], dim=1)

        # out = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "pixel_values": enc["pixel_values"],
        #     "labels": labels,
        # }
        # # stop = input(out)
        # # for item in out["input_ids"]:
        # #     stop = input(item)
        # #     stop = input(self.processor.tokenizer.decoder(item))
        # return out
