"""
Command Line Interface (CLI) for llmasm.
"""

import argparse
import sys
import time
from pathlib import Path

from . import __version__, build_c_library, export_to_asmbin, find_compiled_lib, load_model


def handle_run(args):
    """Executes a single prompt."""
    print(f"Loading model '{args.model}' (backend: {args.backend})...")
    model = load_model(args.model, backend=args.backend)
    info = model.info
    print(f"Loaded: dim={info['dim']} layers={info['n_layers']} heads={info['n_heads']}")
    print(f"\n--- Output ---")

    if args.stream:
        for chunk in model.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temp, stream=True):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()
    else:
        t0 = time.perf_counter()
        res = model.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temp, stream=False)
        elapsed = time.perf_counter() - t0
        text = res if isinstance(res, str) else res.text
        print(text)
        print(f"\n[Generated in {elapsed:.2f}s]")


def handle_chat(args):
    """Starts interactive chat session."""
    print(f"Loading model '{args.model}' for chat...")
    model = load_model(args.model, backend=args.backend)
    print("Interactive Chat Session. Type 'exit' or 'quit' to quit.\n")

    while True:
        try:
            user_msg = input("\nUser > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if user_msg.lower() in ("exit", "quit", "q"):
            break

        sys.stdout.write("Assistant > ")
        sys.stdout.flush()

        for chunk in model.generate(user_msg, max_tokens=args.max_tokens, temperature=args.temp, stream=True):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()


def handle_convert(args):
    """Converts a HuggingFace model directory to .asmbin."""
    print(f"Exporting HuggingFace model '{args.model_dir}' -> '{args.out}'...")
    export_to_asmbin(
        model_dir=args.model_dir,
        out_path=args.out,
        max_seq_len=args.max_seq_len,
        scale_dtype=args.scale_dtype,
        progress_callback=lambda msg, p: print(f"[{int(p*100):3d}%] {msg}"),
    )
    print("Export completed successfully!")


def handle_info(args):
    """Displays metadata and architecture info of .asmbin model."""
    model = load_model(args.model, backend="numpy")
    info = model.info
    print(f"\n=== Model Information: {args.model} ===")
    for k, v in info.items():
        print(f"  {k:15s}: {v}")
    print()


def handle_build_c(args):
    """Builds the native C shared library."""
    print("Compiling C inference engine (src_c/llmasm_engine.c)...")
    res = build_c_library(verbose=True)
    if res:
        print(f"Build succeeded: {res}")
    else:
        print("Build failed. Ensure a C compiler (gcc, clang, or cl) is in your PATH.")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="llmasm",
        description="llmasm: Python library and high-performance C runtime for llm.asm models",
    )
    parser.add_argument("-v", "--version", action="version", version=f"llmasm {__version__}")
    subparsers = parser.add_subparsers(dest="subcommand")

    # Run
    p_run = subparsers.add_parser("run", help="Generate completion for a prompt")
    p_run.add_argument("-m", "--model", required=True, help="Path to .asmbin model file")
    p_run.add_argument("-p", "--prompt", required=True, help="Prompt text")
    p_run.add_argument("-n", "--max-tokens", type=int, default=64, help="Max tokens to generate")
    p_run.add_argument("-t", "--temp", type=float, default=0.0, help="Sampling temperature")
    p_run.add_argument("--backend", choices=["auto", "c", "numpy"], default="auto", help="Inference backend")
    p_run.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming output")

    # Chat
    p_chat = subparsers.add_parser("chat", help="Start interactive multi-turn chat")
    p_chat.add_argument("-m", "--model", required=True, help="Path to .asmbin model file")
    p_chat.add_argument("-n", "--max-tokens", type=int, default=128, help="Max tokens per response")
    p_chat.add_argument("-t", "--temp", type=float, default=0.0, help="Sampling temperature")
    p_chat.add_argument("--backend", choices=["auto", "c", "numpy"], default="auto", help="Inference backend")

    # Convert
    p_conv = subparsers.add_parser("convert", help="Convert HuggingFace safetensors to .asmbin")
    p_conv.add_argument("--model-dir", required=True, help="Directory containing config.json, tokenizer.json, and safetensors")
    p_conv.add_argument("--out", required=True, help="Output .asmbin path")
    p_conv.add_argument("--max-seq-len", type=int, default=4096, help="Maximum context sequence length")
    p_conv.add_argument("--scale-dtype", choices=["fp16", "fp32"], default="fp16", help="Block scale dtype (v2 fp16 default)")

    # Info
    p_info = subparsers.add_parser("info", help="Inspect .asmbin model header and architecture")
    p_info.add_argument("-m", "--model", required=True, help="Path to .asmbin model file")

    # Build C
    p_build = subparsers.add_parser("build-c", help="Compile native C shared library (libllmasm)")

    args = parser.parse_args(argv)
    if not args.subcommand:
        parser.print_help()
        return

    if args.subcommand == "run":
        handle_run(args)
    elif args.subcommand == "chat":
        handle_chat(args)
    elif args.subcommand == "convert":
        handle_convert(args)
    elif args.subcommand == "info":
        handle_info(args)
    elif args.subcommand == "build-c":
        handle_build_c(args)


if __name__ == "__main__":
    main()
