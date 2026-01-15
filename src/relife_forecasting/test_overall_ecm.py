#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_script(path: str) -> int:
    """Esegue uno script Python e ritorna il codice di uscita."""
    script_path = Path(path)

    if not script_path.exists():
        print(f"[ERRORE] Script non trovato: {script_path}")
        return 1

    print(f"\n=== Avvio {script_path} ===")
    # Usa lo stesso interprete con cui stai eseguendo questo file
    result = subprocess.run([sys.executable, str(script_path)])
    print(f"=== Fine {script_path} (returncode={result.returncode}) ===\n")
    return result.returncode

def main():
    # 1) Primo script
    rc1 = run_script("ecm_application.py")
    if rc1 != 0:
        print("[STOP] Il primo script è terminato con errore, non lancio il secondo.")
        sys.exit(rc1)

    # 2) Secondo script (solo se il primo è andato bene)
    rc2 = run_script("report_multiple_simulation.py")
    if rc2 != 0:
        print("[WARN] Il secondo script è terminato con errore.")
    else:
        print("[OK] Entrambi gli script sono stati eseguiti correttamente.")

    # sys.exit(rc2)

if __name__ == "__main__":
    main()
