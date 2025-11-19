@torch.no_grad()
def inference_step(self, batch, prompt_mask=None, cls_features=None):
    self.model.eval()
    
    input_ids = batch['input_ids']
    attn_masks = batch['attention_mask']
    inputs_embeds = self.embed_tokens(input_ids)

    # === 1. 计算 embedding key ===
    if self.embedding_key == 'mean':
        inputs_embeds_mean = torch.mean(inputs_embeds, dim=1)
    elif self.embedding_key == 'max':
        inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0]
    elif self.embedding_key == 'mean_max':
        inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0] + 2 * torch.mean(inputs_embeds, dim=1)
    elif self.embedding_key == 'cls':
        if cls_features is None:
            inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0]
        else:
            inputs_embeds_mean = cls_features
    else:
        raise NotImplementedError("Not supported way of calculating embedding keys!")

    # === 2. 选择 prompt ===
    prompt_norm = self.l2_normalize(self.prompt_key, dim=1).to("cuda")  # (Pool_size, C)
    inputs_embeds_norm = self.l2_normalize(inputs_embeds_mean, dim=1)   # (B, C)

    similarity = torch.matmul(inputs_embeds_norm, prompt_norm.t())      # (B, Pool_size)

    if prompt_mask is None:
        _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # (B, top_k)
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([
                    prompt_id,
                    torch.full((self.pool_size - prompt_id.shape[0],),
                               torch.min(idx.flatten()),
                               device=prompt_id.device)
                ])
                id_counts = torch.cat([
                    id_counts,
                    torch.full((self.pool_size - id_counts.shape[0],),
                               0,
                               device=id_counts.device)
                ])
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            major_prompt_id = prompt_id[major_idx]
            idx = major_prompt_id.expand(inputs_embeds.shape[0], -1)  # (B, top_k)
    else:
        idx = prompt_mask  # (B, top_k)

    # === 3. 拼接 prompt ===
    batched_prompt_raw = self.model.model.prompt[idx]                  # (B, top_k, length, C)
    batch_size, top_k, length, c = batched_prompt_raw.shape
    batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)  # (B, top_k*length, C)

    # 拼接到输入 embeddings 前面
    inputs_embeds = torch.cat([batched_prompt, inputs_embeds], axis=1)  # (B, prefix+seq, C)

    # === 4. 扩展 attention mask ===
    prefix_length = batched_prompt.shape[1]
    attn_masks = torch.cat(
        (torch.ones(batch_size, prefix_length, device=attn_masks.device, dtype=attn_masks.dtype),
         attn_masks),
        dim=1
    )

    # === 5. 前向推理 ===
    outputs = self.model(inputs_embeds=inputs_embeds,
                         attention_mask=attn_masks,
                         use_cache=False)

    return outputs  # e.g. logits, hidden states
