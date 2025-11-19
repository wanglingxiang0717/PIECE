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
    image_grid_thw: torch.Tensor 

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
        # messages = [
        #         {"role": "user",
        #         "content": [
        #             {"type": "text", "text": q_text},
        #             {"type": "image"}
        #         ]},
        # ]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": q_text},
                ],
            }
        ]
        # prompt = self.processor.tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )  
        # prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        # inputs = self.processor(raw_image, prompt, return_tensors="pt")

        a_input_ids = self.processor.tokenizer(
            a_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        # stop = input(inputs)
        res = QaImageOutput(
            q_input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            a_input_ids=a_input_ids,
            image_grid_thw=inputs.get("image_grid_thw")
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
        image_grid_thw = []
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
            image_grid_thw.append(qaimage_output.image_grid_thw)
        
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
        final_image_grid_thw = torch.concat(image_grid_thw, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        out = {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "image_grid_thw": final_image_grid_thw,
            "attention_mask": attention_mask,
        }
        # stop = input(out)
        return out
