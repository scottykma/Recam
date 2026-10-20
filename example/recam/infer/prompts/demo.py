"""Demo prompt set sampled from the training corpus."""

PROMPT_CONFIGS = {
    "demo_person_lake": {
        "dense_prompt": (
            "A woman in a white shirt walks along a lakeside promenade, with "
            "mountains across the water."
        ),
        "ref_images": {"0": "assets/demo/person_lake.png"},
    },
    "demo_cat": {
        "dense_prompt": (
            "A gray cat peeks out of a brown paper bag, its yellow eyes "
            "catching the light."
        ),
        "ref_images": {"0": "assets/demo/cat.png"},
    },
}
