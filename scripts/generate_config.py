import argparse
from pathlib import Path
from jinja2 import Template

def render_config(nx: int, ny: int, template_path: str) -> str:
    with open(template_path, "r") as f:
        template = Template(f.read())
    return template.render(nx=nx, ny=ny)

def save_config(rendered_yaml: str, output_path: str):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(rendered_yaml)
    print(f"[✓] Config saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Render WindGym config from Jinja2 template.")
    parser.add_argument("--nx", type=int, required=True, help="Number of turbines in x-direction.")
    parser.add_argument("--ny", type=int, default=1, help="Number of turbines in y-direction.")
    parser.add_argument("--template", type=str, required=True, help="Path to the base Jinja2 config template.")
    parser.add_argument("--output", type=str, default=None, help="Where to save rendered config (YAML).")
    parser.add_argument("--print", action="store_true", help="Print rendered config to stdout instead of saving.")

    args = parser.parse_args()

    if not args.output and not args.print:
        parser.error("You must specify at least one of --output or --print")

    rendered = render_config(args.nx, args.ny, args.template)

    if args.print:
        print("--- Rendered Config ---\n")
        print(rendered)

    if args.output:
        save_config(rendered, args.output)

if __name__ == "__main__":
    main()

