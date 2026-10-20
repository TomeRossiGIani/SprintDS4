import argparse, json
from pathlib import Path
from pipeline import Analyzer

p = argparse.ArgumentParser(description="Analisar transcrição TXT local")
p.add_argument("arquivo", type=Path)
p.add_argument("--saida", type=Path)
args = p.parse_args()
try:
    result = json.dumps(
        Analyzer().analyze(args.arquivo.read_text(encoding="utf-8")), ensure_ascii=False, indent=2
    )
    if args.saida:
        args.saida.write_text(result, encoding="utf-8")
    else:
        print(result)
except (OSError, ValueError) as exc:
    p.exit(1, str(exc) + "\n")
