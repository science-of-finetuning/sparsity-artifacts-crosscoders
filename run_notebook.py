import papermill as pm
from pathlib import Path
from time import time
from argparse import ArgumentParser
import os
import sys
from coolname import generate_slug


def run_notebook(
    notebook: str,
    crosscoder: str,
    remote: bool = False,
    device: str = "auto",
    batch_size: int = 32,
    exp_id: str | None = None,
    layer: int | None = None,
    generate_id: bool = False,
    extra_args: list = [],
) -> None:
    t = time()
    root = Path(__file__).parent
    save_path = root / "results" / notebook
    save_path.mkdir(exist_ok=True, parents=True)

    notebook_root = root / "notebooks"
    source_notebook_path = notebook_root / f"{notebook}.ipynb"

    exp_id = ((str(int(time())) + "_") if generate_id else "") + (
        exp_id or (generate_slug(2) if generate_id else "")
    )

    target_notebook_path = save_path / (
        crosscoder + (f"_{exp_id}" if exp_id else "") + f"_{notebook}.ipynb"
    )

    print(f"Saving to {target_notebook_path}")
    kwargs = {
        "exp_id": exp_id,
        "remote": remote,
        "device": device,
        "batch_size": batch_size,
        "layer": layer,
        "extra_args": extra_args,
        "interactive": False,
        "crosscoder": crosscoder,
    }
    print(f"Running {notebook} with {kwargs}")

    try:
        print(source_notebook_path, target_notebook_path, kwargs)
        pm.execute_notebook(
            source_notebook_path,
            target_notebook_path,
            parameters=kwargs,
            log_output=True,
            stdout_file=sys.stdout,
            stderr_file=sys.stderr,
        )
        print(f"Execution time: {time()-t:.2f}s")
    except (pm.PapermillExecutionError, KeyboardInterrupt) as e:
        print(e)
        if isinstance(e, pm.PapermillExecutionError):
            print("Error in notebook")
        delete = input(f"Delete notebook {target_notebook_path} ? (y/n)")
        if delete.lower() == "y":
            target_notebook_path.unlink()
        else:
            print(f"Notebook saved")
        exit(1)
    print(f"Notebook succesfully executed and saved to\n{target_notebook_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    parser = ArgumentParser()
    parser.add_argument("notebook", type=str)
    parser.add_argument("--crosscoder", "-c", type=str)
    parser.add_argument(
        "--remote",
        default=False,
        action="store_true",
        help="Use ndif remote execution. Check https://nnsight.net/status for available models",
    )
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--exp-id")
    parser.add_argument("--generate-id", action="store_true")
    parser.add_argument("--layer", type=int)
    args, unknown = parser.parse_known_args()

    run_notebook(
        notebook=args.notebook,
        crosscoder=args.crosscoder,
        remote=args.remote,
        device=args.device,
        batch_size=args.batch_size,
        exp_id=args.exp_id,
        layer=args.layer,
        generate_id=args.generate_id,
        extra_args=unknown,
    )
