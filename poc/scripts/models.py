"""Endpoint registry. est_cost is per image for a ~1MP square output with 3 input
images, taken from each fal model page on 2026-11-11 (GPT is token-billed, so
its figure is an estimate)."""

SQUARE = {"width": 1024, "height": 1024}

MODELS = {
    "nano_banana_pro": dict(
        endpoint="fal-ai/nano-banana-pro/edit", est_cost=0.15, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, aspect_ratio="1:1", resolution="1K", output_format="png"),
    ),
    "nano_banana_2": dict(
        endpoint="fal-ai/nano-banana-2/edit", est_cost=0.08, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, aspect_ratio="1:1", output_format="png"),
    ),
    "gpt_image_2_high": dict(
        endpoint="openai/gpt-image-2/edit", est_cost=0.20, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, image_size=SQUARE, quality="high", output_format="png"),
    ),
    "gpt_image_2_5_flare": dict(
        endpoint="openai/gpt-image-2.5/flare/edit", est_cost=0.20, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, image_size=SQUARE, quality="high", output_format="png"),
    ),
    "gpt_image_2_5_sunburst": dict(
        endpoint="openai/gpt-image-2.5/sunburst/edit", est_cost=0.20, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, image_size=SQUARE, quality="high", output_format="png"),
    ),
    "seedream_5_pro": dict(
        endpoint="bytedance/seedream/v5/pro/edit", est_cost=0.0765, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, image_size=SQUARE, output_format="png"),
    ),
    "seedream_4_5": dict(
        endpoint="fal-ai/bytedance/seedream/v4.5/edit", est_cost=0.04, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, image_size={"width": 2048, "height": 2048}),
    ),
    "flux_2_pro": dict(
        endpoint="fal-ai/flux-2-pro/edit", est_cost=0.075, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, image_size=SQUARE, output_format="png"),
    ),
    "flux_kontext_max": dict(
        endpoint="fal-ai/flux-pro/kontext/max/multi", est_cost=0.08, license="API only",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, aspect_ratio="1:1", output_format="png"),
    ),
    # open-weight models: these are the ones we can self-host on 4090/5090
    "qwen_edit_2511": dict(
        endpoint="fal-ai/qwen-image-edit-2511", est_cost=0.03, license="Apache-2.0 (self-hostable)",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, image_size=SQUARE, output_format="png"),
    ),
    "flux_2_klein_9b": dict(
        endpoint="fal-ai/flux-2/klein/9b/edit", est_cost=0.044, license="non-commercial weights (self-host needs BFL licence)",
        args=lambda p, urls: dict(prompt=p, image_urls=urls, image_size=SQUARE, num_inference_steps=8, output_format="png"),
    ),
}
