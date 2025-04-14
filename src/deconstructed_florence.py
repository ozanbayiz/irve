import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor

class FlorenceVisionEncoder(nn.Module):
    """
    Encapsulates the vision encoding part of Florence-2, allowing for intermediate feature extraction.
    """
    def __init__(
        self,
        vision_tower,
        image_pos_embed, 
        visual_temporal_embed, 
        image_projection, 
        image_proj_norm, 
        vision_config, 
        device, 
        dtype
    ):
        super().__init__()
        # Store references to the relevant layers/parameters from the original model
        self.conv_embeds = vision_tower.convs
        self.blocks = vision_tower.blocks
        self.pos_embed = image_pos_embed
        self.temporal_embed = visual_temporal_embed
        self.projection_layer = image_projection
        self.proj_norm = image_proj_norm
        self.config = vision_config
        self.device = device
        self.dtype = dtype

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input pixel values into image features.
        """
        final_features, _ = self._encode_internal(pixel_values, return_intermediates=False)
        return final_features

    def encode_with_intermediate_outputs(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Encodes the input pixel values and returns both final features and intermediate block outputs.
        """
        return self._encode_internal(pixel_values, return_intermediates=True)

    def _encode_internal(
        self,
        pixel_values: torch.Tensor,
        return_intermediates: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Internal encoding logic, replicating Florence2ForConditionalGeneration._encode_image.
        """
        # Input validation and preparation (simplified)
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values to be 4D (B, C, H, W), got {pixel_values.ndim}D")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        batch_size, _, H, W = pixel_values.shape
        T = 1 # Assuming single frame input for simplicity based on _encode_image
        current_size = (H, W)
        x = pixel_values
        intermediate_outputs = []

        # Pass through convolutional embeddings and blocks
        for i in range(len(self.blocks)):
            x, current_size = self.conv_embeds[i](x, current_size)
            x, current_size = self.blocks[i](x, current_size)
            if return_intermediates:
                intermediate_outputs.append(x.clone())


        # Apply positional embeddings

        ### NOTE: old
        # if self.pos_embed is not None:
        #     x = x.view(batch_size * T, -1, x.shape[-1])
        #     num_tokens = x.shape[-2]
        #     h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
        #     if h * w != num_tokens:
        #         raise ValueError('Feature map aspect ratio not supported for pos embed')
        #     x = x.view(batch_size * T, h, w, x.shape[-1])
        #     pos_embed = self.pos_embed(x) # LearnedAbsolutePositionEmbedding2D expects (B, H, W, C)
        #     x = x + pos_embed
        #     x = x.view(batch_size, T * h * w, x.shape[-1]) # Reshape back

        ### NOTE: taken from original
        if self.pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
            assert h * w == num_tokens, 'only support square feature maps for now'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h*w, x.shape[-1])


        # Apply temporal embeddings (simplified for T=1)
        ### NOTE: old
        # if self.temporal_embed is not None and T > 1:
        #      # Simplified: Original logic is slightly more complex for T>1
        #     temporal_embed = self.temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
        #     x = x.view(batch_size, T, -1, x.shape[-1]) + temporal_embed.view(1, T, 1, x.shape[-1])
        
        ### NOTE: taken from original
        if self.temporal_embed is not None:
            temporal_embed = self.temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + temporal_embed.view(1, T, 1, x.shape[-1])


        # Feature selection based on config
        # Replicating logic from _encode_image
        ### NOTE: old
        # x_feat_dict = {}
        # # Reshape x to (B, T, num_tokens, C) before pooling if T > 1, else keep (B, num_tokens, C)
        # if T > 1:
        #      x_for_pooling = x.view(batch_size, T, -1, x.shape[-1])
        #      spatial_avg_pool_x = x_for_pooling.mean(dim=2) # (B, T, C)
        #      temporal_avg_pool_x = x_for_pooling.mean(dim=1) # (B, num_tokens, C)
        #      last_frame_x = x_for_pooling[:, -1] # (B, num_tokens, C)
        # else:
        #      # If T=1, x is already (B, num_tokens, C)
        #      spatial_avg_pool_x = x.mean(dim=1, keepdim=True) # (B, 1, C), keep dim for compatibility? Check original. No, should be (B, C)
        #      spatial_avg_pool_x = x.mean(dim=1) # (B, C)
        #      temporal_avg_pool_x = x # (B, num_tokens, C) as temporal average is just the single frame
        #      last_frame_x = x # (B, num_tokens, C)

        # x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x
        # x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x
        # x_feat_dict['last_frame'] = last_frame_x

        ### NOTE: taken from original
        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for source in self.config.image_feature_source:
            if source not in x_feat_dict:
                raise ValueError(f"Invalid image feature source: {source}")
            # Need to handle shapes carefully here. Original model selects then concatenates.
            # Example: if source is ['last_frame'], new_x gets [(B, num_tokens, C)]
            # if source is ['spatial_avg_pool'], new_x gets [(B, C)] -> needs view/unsqueeze before cat maybe?
            # Let's assume for now it's typically ['last_frame'] or similar structure leading to (B, seq_len, C)
            feature_tensor = x_feat_dict[source]
            # Ensure feature_tensor is 3D for concatenation B, Seq, Dim
            # if feature_tensor.ndim == 2:
                #  feature_tensor = feature_tensor.unsqueeze(1) # Add sequence dimension if needed, e.g., for spatial_avg_pool

            new_x.append(feature_tensor)

        if not new_x:
             raise ValueError("No image features selected based on config")

        # Concatenate selected features along the sequence dimension
        x = torch.cat(new_x, dim=1) # Concatenate along sequence dim

        # Apply projection and normalization
        # Original uses Parameter, matmul is appropriate
        x = x @ self.projection_layer
        image_features = self.proj_norm(x)

        return image_features, intermediate_outputs
    
    def get_latent_representations(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns the outputs of all blocks in the vision tower.
        """
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values to be 4D (B, C, H, W), got {pixel_values.ndim}D")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        batch_size, _, H, W = pixel_values.shape
        current_size = (H, W)
        x = pixel_values
        intermediate_outputs = []

        # Pass through convolutional embeddings and blocks
        for i in range(len(self.blocks)):
            x, current_size = self.conv_embeds[i](x, current_size)
            x, current_size = self.blocks[i](x, current_size)
            intermediate_outputs.append(x.clone())

        return intermediate_outputs


class DeconstructedFlorence2:
    """
    Orchestrator class that loads Florence-2 and allows explicit vision encoding.
    """
    def __init__(self, model_name_or_path, device='cuda', dtype=torch.bfloat16, trust_remote_code=True):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            # torch_dtype=dtype # Let from_pretrained handle dtype based on saved weights initially
        ).to(device).to(dtype) # Move and cast after loading

        self.model.eval() # Set to evaluation mode

        self.vision_config = self.model.config.vision_config
        self.text_config = self.model.config.text_config
        self.device = device
        self.dtype = dtype

        # Instantiate the vision encoder
        self.vision_encoder = FlorenceVisionEncoder(
            vision_tower=self.model.vision_tower,
            image_pos_embed=self.model.image_pos_embed,
            visual_temporal_embed=self.model.visual_temporal_embed,
            # image_pos_embed=getattr(self.model, 'image_pos_embed', None), # Use getattr for safety
            # visual_temporal_embed=getattr(self.model, 'visual_temporal_embed', None),
            image_projection=self.model.image_projection,
            image_proj_norm=self.model.image_proj_norm,
            vision_config=self.vision_config,
            device=self.device,
            dtype=self.dtype
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encodes image using the decoupled vision encoder.
        """
        with torch.no_grad():
             image_features = self.vision_encoder.encode(pixel_values)
        return image_features

    def encode_image_with_intermediates(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Encodes image using the decoupled vision encoder and returns intermediate outputs.
        """
        with torch.no_grad():
            final_features, intermediate_outputs = self.vision_encoder.encode_with_intermediate_outputs(pixel_values)
        return final_features, intermediate_outputs

    def generate(
        self, 
        input_ids: torch.Tensor, 
        pixel_values: torch.Tensor, 
        **generation_kwargs
    ):
        """
        Generates text conditioned on the image, performing vision encoding explicitly first.
        """
        # 1. Encode image explicitly
        with torch.no_grad():
            image_features = self.vision_encoder.encode(pixel_values) # (B, img_seq_len, D)

            # 2. Embed text prompt
            # Ensure input_ids are on the correct device
            input_ids = input_ids.to(self.device)
            text_embeds = self.model.get_input_embeddings()(input_ids) # (B, text_seq_len, D)

            # 3. Prepare combined inputs for the language model's generate function
            inputs_embeds, attention_mask = self.model._merge_input_ids_with_image_features(image_features, text_embeds)
            
            # Replicate the logic of _merge_input_ids_with_image_features

            # image_token_length = image_features.size(1)
            # text_token_length = text_embeds.size(1)

            # image_attention_mask = torch.ones(image_features.size(0), image_token_length, device=self.device, dtype=torch.long)
            # text_attention_mask = torch.ones(text_embeds.size(0), text_token_length, device=self.device, dtype=torch.long)

            # # Concatenate image features and text embeddings
            # inputs_embeds = torch.cat([image_features, text_embeds], dim=1) # (B, img_seq_len + text_seq_len, D)
            # # Concatenate attention masks
            # attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1) # (B, img_seq_len + text_seq_len)

        # 4. Call language model generate
        # The original model.generate calls language_model.generate internally after merging.
        # We pass the explicitly computed inputs_embeds and attention_mask.
        # input_ids=None because we are providing inputs_embeds.
        generated_ids = self.model.language_model.generate(
            input_ids=None, # We provide embeddings
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, # Pass the combined mask
            **generation_kwargs
        )

        return generated_ids


# Example Usage (Optional - requires model download and an image)
if __name__ == '__main__':
    from PIL import Image
    import requests
    import sys # For exiting cleanly

    # Load model and processor
    model_id = "microsoft/Florence-2-base" # Use base for faster testing
    # processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) # Original processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32 # bfloat16 recommended for GPU

    print(f"Using device: {device}, dtype: {dtype}")

    # Instantiate the deconstructed model
    try:
        deconstructed_model = DeconstructedFlorence2(model_id, device=device, dtype=dtype, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Deconstructed model and processor loaded.")
    except Exception as e:
        print(f"Error loading deconstructed model or processor: {e}", file=sys.stderr)
        exit(1)

    # Load the original model separately using the correct class for comparison and _encode_image access
    try:
        # original_model_for_comparison = Florence2ForConditionalGeneration.from_pretrained(
        #     model_id, trust_remote_code=True
        # ).to(device).to(dtype)
        original_model_for_comparison = deconstructed_model.model
        original_model_for_comparison.eval()
        print("Original comparison model loaded.")
    except Exception as e:
        print(f"Error loading original comparison model: {e}", file=sys.stderr)
        exit(1)


    # Prepare inputs
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        print("Image loaded.")
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        exit(1)

    prompt = "<CAPTION>"

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        # Keep inputs on CPU initially, move to device inside methods/calls
        pixel_values = inputs['pixel_values']
        input_ids = inputs['input_ids']
        print("Inputs processed.")
    except Exception as e:
        print(f"Error processing inputs: {e}", file=sys.stderr)
        exit(1)


    # --- Test Vision Encoder Output ---
    print("\nTesting Vision Encoder equivalence...")
    deconstructed_features = None
    original_features = None
    vision_test_passed = False
    try:
        # 1. Get features from deconstructed encoder
        deconstructed_features = deconstructed_model.encode_image(pixel_values)
        print("  - Deconstructed features generated.")

        # 2. Get features from original model's internal method
        #    Ensure pixel_values are on the correct device/dtype for this call
        with torch.no_grad():
             original_features = original_model_for_comparison._encode_image(
                 pixel_values.to(device=device, dtype=dtype)
             )
        print("  - Original features generated.")

        # 3. Compare outputs
        # Use appropriate tolerances based on dtype
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        if deconstructed_features is not None and original_features is not None:
            if torch.allclose(deconstructed_features, original_features, atol=atol, rtol=rtol):
                print(f"  SUCCESS: Vision encoder outputs match within tolerance (atol={atol}, rtol={rtol}).")
                vision_test_passed = True
            else:
                print(f"  FAILURE: Vision encoder outputs DO NOT match within tolerance (atol={atol}, rtol={rtol}).")
                # Optional: print difference stats
                diff = torch.abs(deconstructed_features - original_features)
                print(f"    Max absolute difference: {diff.max().item()}")
                print(f"    Mean absolute difference: {diff.mean().item()}")
        else:
            print("  Comparison skipped as one or both feature sets could not be generated.")

    except Exception as e:
        print(f"Error during vision encoder comparison: {e}", file=sys.stderr)
        print(f"  Deconstructed features shape: {deconstructed_features.shape if deconstructed_features is not None else 'N/A'}")
        print(f"  Original features shape: {original_features.shape if original_features is not None else 'N/A'}")


    # --- Test Intermediate Outputs (Optional, keep from previous examples) ---
    # print("\nTesting encode_image_with_intermediates...")
    # ...

    # --- Test Generation ---
    print("\nTesting generate...")
    generation_kwargs = {
        "max_new_tokens": 50,
        "num_beams": 3,
        "do_sample": False # For reproducibility
    }

    decoded_text = "[Deconstructed generation failed]"
    deconstructed_ids = None
    try:
        deconstructed_ids = deconstructed_model.generate(
            input_ids=input_ids, # Pass tensors, method handles device placement
            pixel_values=pixel_values,
            **generation_kwargs
        )
        decoded_text = processor.batch_decode(deconstructed_ids, skip_special_tokens=True)[0]
        print("Deconstructed Generation Output:", decoded_text)
    except Exception as e:
        print(f"Error during deconstructed generation: {e}", file=sys.stderr)


    # --- Compare with original model generation ---
    print("\nComparing with original model generation...")
    original_decoded_text = "[Original generation failed]"
    original_ids = None
    try:
        # Re-use the loaded original_model_for_comparison
        hf_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=device, dtype=dtype)

        with torch.no_grad():
            original_ids = original_model_for_comparison.generate(**hf_inputs, **generation_kwargs)

        original_decoded_text = processor.batch_decode(original_ids, skip_special_tokens=True)[0]
        print("Original Generation Output:", original_decoded_text)

        if deconstructed_ids is not None and original_ids is not None:
            if torch.equal(deconstructed_ids, original_ids):
                print("\nSUCCESS: Generated token IDs match!")
            else:
                print("\nWARNING: Generated token IDs do NOT match.")
                print(f"  Deconstructed IDs ({deconstructed_ids.shape}): {deconstructed_ids.tolist()}")
                print(f"  Original IDs ({original_ids.shape}): {original_ids.tolist()}")
                if decoded_text == original_decoded_text:
                     print("However, the decoded text outputs are identical.")
                else:
                     print("And the decoded text outputs also differ.")
        else:
             print("\nComparison skipped as one or both generations failed.")

    except Exception as e:
        print(f"Error during original model generation or comparison: {e}", file=sys.stderr)