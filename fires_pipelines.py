from src.components.combined_columns import load_data
import kfp
from kfp.v2.dsl import pipeline

@pipeline(
    name="fires-pipeline",
    description="Pipeline de ejemplo que llama a load_data"
)
def fires_pipeline(
    project_id: str = "mi-proyecto",
    start_index: int = 1
):
    load_data_task = load_data(
        project_id=project_id,
        start_index=start_index
    )


if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=fires_pipeline,
        package_path="fires_pipeline.json"
    )